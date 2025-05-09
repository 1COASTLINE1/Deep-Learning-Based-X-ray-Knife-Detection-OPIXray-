import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import cv2 # Import OpenCV
from torchvision.ops import nms # Import NMS
import torch.nn as nn # Import nn for wrapper

# Imports for Grad-CAM
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# Import necessary components from your project
from dataloader import KnifeDataset, Resizer, Normalizer, NUM_CLASSES, CLASS_MAP, UnNormalizer
from retinanet import model  # Assuming your model definition is here
from torchvision import transforms

# --- Helper Functions (adapted from csv_eval.py & visualize.py) ---

def draw_caption(image, box, caption):
    """Draws a caption above the box.
       Helper function from visualize.py"""
    b = np.array(box).astype(int)
    # Use a smaller font and thickness for potentially crowded images
    cv2.putText(image, caption, (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2) # Black background
    cv2.putText(image, caption, (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1) # White text

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float, detection boxes
    b: (K, 4) ndarray of float, annotation boxes
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).
    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# --- Wrapper for Grad-CAM ---
class RetinaNetCAMWrapper(nn.Module):
    def __init__(self, model):
        super(RetinaNetCAMWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Return only scores, as EigenCAM relies on hooked activations,
        # and this avoids the list processing error.
        scores = self.model(x)
        return scores
# --- End Wrapper ---

def _get_detections_and_vis_data(dataset, model, score_threshold=0.05, max_detections=100, device='cuda', save_path=None, cam_indices=None, index=None):
    """
    Gets detections for a SINGLE image and prepares visualization data (image tensor, CAM).
    Args:
        ... (same as before) ...
        index (int): The index of the image in the dataset.
    Returns:
        Tuple: (boxes_nms, scores_nms, labels_nms, img_tensor_orig_shape, grayscale_cam)
               Returns None for grayscale_cam if not generated.
               Returns empty arrays for boxes/scores/labels if no detections.
    """
    model.eval()
    unnormalizer = UnNormalizer()
    cam = None
    use_gpu = (device == torch.device("cuda:0") or device=='cuda')
    generate_cam_for_this_image = False

    # --- Grad-CAM Setup (only if needed for this index) ---
    if save_path and cam_indices and index in cam_indices:
        try:
            model_for_cam = model # Using original model as per user changes
            model_for_cam.eval()
            target_layers = [model.layer4[-1]]
            cam = EigenCAM(model=model_for_cam, target_layers=target_layers)
            generate_cam_for_this_image = True # Tentatively enable
            # print(f"Debug: CAM initialized for index {index}") # Optional debug
        except Exception as e:
            print(f"Warning: Error initializing CAM for index {index}. Disabling CAM. Error: {e}")
            cam = None
            generate_cam_for_this_image = False
    # --- End Grad-CAM Setup ---

    # --- Get Data and Run Inference ---
    data = dataset[index]
    scale = data['scale']
    img_tensor_orig_shape = torch.from_numpy(data['img']).permute(2, 0, 1).float() # Keep C, H, W
    img_tensor_batch = img_tensor_orig_shape.unsqueeze(dim=0).to(device)

    with torch.no_grad():
        scores_batch, labels_batch, boxes_batch = model(img_tensor_batch)

    # --- Process Detections ---
    boxes_nms = np.zeros((0, 4))
    scores_nms = np.zeros((0,))
    labels_nms = np.zeros((0,), dtype=np.int32)
    grayscale_cam = None

    if len(boxes_batch) > 0 and boxes_batch[0].shape[0] > 0:
        scores = scores_batch.cpu().numpy()
        labels = labels_batch.cpu().numpy()
        boxes = boxes_batch.cpu().numpy() # These are UN-scaled

        indices_draw = np.where(scores > score_threshold)[0]
        if indices_draw.shape[0] > 0:
            scores_draw = scores[indices_draw]
            labels_draw = labels[indices_draw]
            boxes_draw = boxes[indices_draw]

            sort_draw_idx = np.argsort(-scores_draw)[:max_detections]
            scores_draw = scores_draw[sort_draw_idx]
            labels_draw = labels_draw[sort_draw_idx]
            boxes_draw = boxes_draw[sort_draw_idx]

            # NMS
            if boxes_draw.shape[0] > 0:
                boxes_tensor = torch.from_numpy(boxes_draw).float()
                scores_tensor = torch.from_numpy(scores_draw).float()
                labels_tensor = torch.from_numpy(labels_draw).int()
                keep_indices_all = []
                nms_iou_threshold = 0.5
                unique_labels = torch.unique(labels_tensor)
                for label_val in unique_labels:
                    class_mask = (labels_tensor == label_val)
                    keep = nms(boxes_tensor[class_mask], scores_tensor[class_mask], nms_iou_threshold)
                    original_indices = torch.where(class_mask)[0]
                    keep_indices_all.extend(original_indices[keep].tolist())
                keep_indices_all = sorted(list(set(keep_indices_all)))
                boxes_nms = boxes_draw[keep_indices_all]
                scores_nms = scores_draw[keep_indices_all]
                labels_nms = labels_draw[keep_indices_all]

    # --- Generate CAM (if conditions met) ---
    if generate_cam_for_this_image and boxes_nms.shape[0] > 0 and cam is not None:
        try:
            # targets = [ClassifierOutputTarget(1)] # EigenCAM doesn't need targets like this
            grayscale_cam_batch = cam(input_tensor=img_tensor_batch) # removed targets
            grayscale_cam = grayscale_cam_batch[0, :]
        except Exception as e:
            print(f"Warning: Error generating CAM for image {index}. CAM disabled for this image. Error: {e}")
            # import traceback # Optional for deeper debug
            # traceback.print_exc() # Optional for deeper debug
            grayscale_cam = None # Ensure it's None on error

    # Return NMS results, original tensor, and CAM map (or None)
    # Note: boxes_nms are UN-scaled (relative to transformed image size)
    return boxes_nms, scores_nms, labels_nms, img_tensor_orig_shape, grayscale_cam, scale

def _get_annotations(dataset):
    """ Get ground truth annotations from the dataset.
     Returns:
        all_annotations: list[list[np.ndarray(M, 4)]], outer list is image index,
                        inner list is class label, ndarray is [x1, y1, x2, y2]
    """
    num_images = len(dataset)
    all_annotations = [[None for _ in range(dataset.num_classes())] for _ in range(num_images)]

    print("Getting annotations...")
    for index in tqdm(range(num_images)):
        # Load annotations for the image: numpy array (M, 5) [x1, y1, x2, y2, class_id]
        annotations = dataset.load_annotations(index)

        # Store annotations per class
        for label in range(dataset.num_classes()):
            class_annotations = annotations[annotations[:, 4] == label, :4].copy()
            all_annotations[index][label] = class_annotations

    return all_annotations

def evaluate(dataset, model, iou_threshold=0.5, score_threshold=0.05, max_detections=100, device='cuda', save_path=None, cam_indices=None):
    """ Evaluate a dataset using the model, performing detailed matching per image."""

    num_dataset_images = len(dataset)
    unnormalizer = UnNormalizer()

    # Data structures for metrics
    tp_fp_scores_per_class = {label: {'tp': [], 'fp': [], 'scores': []} for label in range(dataset.num_classes())}
    num_annotations_per_class = {label: 0 for label in range(dataset.num_classes())}
    images_with_good_iou_overall = set()
    per_class_images_with_good_iou = {label: set() for label in range(dataset.num_classes())}
    images_containing_class = {label: set() for label in range(dataset.num_classes())}

    # Setup save directories
    save_correct_path = None
    save_incorrect_path = None
    if save_path:
        save_correct_path = os.path.join(save_path, "correct_prediction")
        save_incorrect_path = os.path.join(save_path, "incorrect_prediction")
        os.makedirs(save_correct_path, exist_ok=True)
        os.makedirs(save_incorrect_path, exist_ok=True)
        print(f"Saving correct predictions to: {save_correct_path}")
        print(f"Saving incorrect predictions to: {save_incorrect_path}")

    print("Processing images and matching detections...")
    for index in tqdm(range(num_dataset_images)):
        # 1. Get Detections & Visualization Data
        boxes_pred_nms, scores_pred_nms, labels_pred_nms, \
        img_tensor_orig, grayscale_cam, scale = _get_detections_and_vis_data(
            dataset, model, score_threshold, max_detections, device, save_path, cam_indices, index
        )
        # Scale predicted boxes back to original image dimensions for IoU calculation
        boxes_pred_nms_scaled = boxes_pred_nms / scale

        # 2. Get Annotations
        annotations_gt = dataset.load_annotations(index) # (M, 5) [x1,y1,x2,y2,label]
        boxes_gt = annotations_gt[:, :4]
        labels_gt = annotations_gt[:, 4].astype(int)
        num_gt_boxes = boxes_gt.shape[0]
        gt_matched = np.zeros(num_gt_boxes, dtype=bool) # Track matched GT boxes

        # Update total annotation count per class AND track images containing the class
        for gt_label in labels_gt:
            if gt_label in num_annotations_per_class:
                num_annotations_per_class[gt_label] += 1
                images_containing_class[gt_label].add(index) # Add image index if it has this GT class

        # 3. Perform Matching
        detections_status = [] # Store {'box', 'score', 'label', 'status': 'TP'/'FP'}
        highest_scoring_fp = {'score': -1, 'label': -1}
        first_fn_label = -1

        # Sort predictions by score (descending) for matching standard practice
        sort_indices = np.argsort(-scores_pred_nms)

        for det_idx in sort_indices:
            pred_box = boxes_pred_nms_scaled[det_idx]
            pred_score = scores_pred_nms[det_idx]
            pred_label = labels_pred_nms[det_idx]
            is_tp = False

            if num_gt_boxes > 0:
                # Find potential GT matches of the same class
                gt_indices_for_class = np.where(labels_gt == pred_label)[0]

                if len(gt_indices_for_class) > 0:
                    overlaps = compute_overlap(np.expand_dims(pred_box, axis=0), boxes_gt[gt_indices_for_class, :])[0]
                    best_overlap_local_idx = np.argmax(overlaps)
                    best_iou = overlaps[best_overlap_local_idx]
                    best_match_gt_global_idx = gt_indices_for_class[best_overlap_local_idx]

                    if best_iou >= iou_threshold and not gt_matched[best_match_gt_global_idx]:
                        # Match found! TP
                        tp_fp_scores_per_class[pred_label]['tp'].append(1)
                        tp_fp_scores_per_class[pred_label]['fp'].append(0)
                        tp_fp_scores_per_class[pred_label]['scores'].append(pred_score)
                        gt_matched[best_match_gt_global_idx] = True
                        is_tp = True
                        images_with_good_iou_overall.add(index)
                        per_class_images_with_good_iou[pred_label].add(index)
                        detections_status.append({'box': boxes_pred_nms[det_idx], 'score': pred_score, 'label': pred_label, 'status': 'TP'})

            if not is_tp:
                # No match found, or matched GT already taken: FP
                tp_fp_scores_per_class[pred_label]['tp'].append(0)
                tp_fp_scores_per_class[pred_label]['fp'].append(1)
                tp_fp_scores_per_class[pred_label]['scores'].append(pred_score)
                detections_status.append({'box': boxes_pred_nms[det_idx], 'score': pred_score, 'label': pred_label, 'status': 'FP'})
                # Track highest scoring FP
                if pred_score > highest_scoring_fp['score']:
                    highest_scoring_fp['score'] = pred_score
                    highest_scoring_fp['label'] = pred_label

        # Identify FNs
        fn_indices = np.where(gt_matched == False)[0]
        if len(fn_indices) > 0 and first_fn_label == -1:
            first_fn_label = labels_gt[fn_indices[0]]

        # 4. Visualize and Save Image (if save_path provided)
        if save_path:
            is_correct = (highest_scoring_fp['label'] == -1 and len(fn_indices) == 0 and len(detections_status) > 0) # Correct only if TPs exist and NO FPs/FNs

            # Prepare base image (unnormalized RGB float 0-1)
            img_unnormalized = unnormalizer(img_tensor_orig.clone())
            rgb_img_float = img_unnormalized.permute(1, 2, 0).cpu().numpy()
            rgb_img_float = np.clip(rgb_img_float, 0, 1)
            vis_image = None

            # Create CAM overlay or base image
            if grayscale_cam is not None:
                 vis_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True) # RGB uint8
            else:
                 vis_image = (rgb_img_float * 255).astype(np.uint8).copy()

            # --- Scale GT boxes to match the visualized image size ---
            boxes_gt_scaled = boxes_gt * scale
            # --- End Scaling ---

            # Draw GT boxes (Red) - using scaled boxes
            for gt_idx in range(num_gt_boxes):
                gt_box = boxes_gt_scaled[gt_idx].astype(int) # Use scaled GT box
                color = (255, 0, 0) # Red for GT
                cv2.rectangle(vis_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=color, thickness=2)

            # Draw Prediction boxes (Green=TP, Yellow=FP)
            for det in detections_status:
                 pred_b = det['box'].astype(int) # Use unscaled box for drawing
                 label_idx = det['label']
                 score = det['score']
                 class_name = dataset.label_to_name(label_idx)
                 caption = f"{class_name}: {score:.2f}"
                 color = (0, 255, 0) if det['status'] == 'TP' else (255, 255, 0) # Green TP, Yellow FP
                 cv2.rectangle(vis_image, (pred_b[0], pred_b[1]), (pred_b[2], pred_b[3]), color=color, thickness=2) # Thicker pred box
                 draw_caption(vis_image, pred_b, caption) # Use helper to draw caption

            # Convert final image to BGR for saving
            final_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

            # Determine filename and save path
            try:
                basename = dataset.image_files_basenames[index]
                filename = f"{basename}.jpg" # Default filename
                save_dir = save_incorrect_path # Default to incorrect

                if is_correct:
                    save_dir = save_correct_path
                    # Find highest scoring TP to use in filename
                    best_tp_score = -1
                    best_tp_label = -1
                    for det in detections_status:
                        if det['status'] == 'TP' and det['score'] > best_tp_score:
                           best_tp_score = det['score']
                           best_tp_label = det['label']
                    if best_tp_label != -1:
                         class_name = dataset.label_to_name(best_tp_label)
                         filename = f"{class_name}_{best_tp_score:.2f}_{basename}.jpg"
                else: # Incorrect
                    error_class_label = highest_scoring_fp['label'] if highest_scoring_fp['label'] != -1 else first_fn_label
                    if error_class_label != -1:
                        class_name = dataset.label_to_name(error_class_label)
                        filename = f"{class_name}_{basename}.jpg" # Use error class name
                    # else filename remains basename.jpg

                output_filepath = os.path.join(save_dir, filename)
                cv2.imwrite(output_filepath, final_image_bgr)

            except Exception as e:
                print(f"Error saving visualization for image {index}: {e}")

    # 5. Calculate Final Metrics
    print("Calculating final metrics...")
    per_class_final_metrics = {}
    total_tp_overall = 0
    total_fp_overall = 0
    total_annotations_overall = sum(num_annotations_per_class.values())

    for label in range(dataset.num_classes()):
        tp = np.array(tp_fp_scores_per_class[label]['tp'])
        fp = np.array(tp_fp_scores_per_class[label]['fp'])
        scores = np.array(tp_fp_scores_per_class[label]['scores'])
        num_annotations = num_annotations_per_class[label]

        if len(scores) == 0:
            per_class_final_metrics[label] = {'AP': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'NumAnnotations': num_annotations, 'IoU_GT_Threshold_Ratio': 0.0}
            continue

        # Sort by score
        indices = np.argsort(-scores)
        tp = tp[indices]
        fp = fp[indices]

        # Compute cumulative TP/FP
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        # Compute recall and precision curves
        recall = tp_cum / num_annotations if num_annotations > 0 else np.zeros_like(tp_cum)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

        # Compute AP
        average_precision = _compute_ap(recall, precision)

        # Calculate final P, R, F1 for the class
        total_tp_class = tp_cum[-1] if len(tp_cum) > 0 else 0
        total_fp_class = fp_cum[-1] if len(fp_cum) > 0 else 0
        recall_class = total_tp_class / num_annotations if num_annotations > 0 else 0.0
        precision_class = total_tp_class / (total_tp_class + total_fp_class) if (total_tp_class + total_fp_class) > 0 else 0.0
        f1_class = 0.0
        if precision_class + recall_class > 0:
            f1_class = 2 * (precision_class * recall_class) / (precision_class + recall_class)

        # Calculate per-class IoU > threshold ratio
        iou_gt_threshold_ratio_class = 0.0
        # Get the count of images actually containing this class
        num_images_with_class = len(images_containing_class[label])
        if num_images_with_class > 0:
             iou_gt_threshold_ratio_class = len(per_class_images_with_good_iou[label]) / num_images_with_class

        per_class_final_metrics[label] = {
            'AP': average_precision,
            'Precision': precision_class,
            'Recall': recall_class,
            'F1': f1_class,
            'NumAnnotations': num_annotations,
            'IoU_GT_Threshold_Ratio': iou_gt_threshold_ratio_class
        }

        # Accumulate for overall metrics
        total_tp_overall += total_tp_class
        total_fp_overall += total_fp_class # Note: Summing FPs across classes isn't standard for overall P/R from AP curves

    # Calculate overall mAP
    mAP = np.mean([metrics['AP'] for metrics in per_class_final_metrics.values() if metrics['NumAnnotations'] > 0])

    # Calculate overall P, R, F1 based on aggregated counts
    overall_precision = total_tp_overall / (total_tp_overall + total_fp_overall) if (total_tp_overall + total_fp_overall) > 0 else 0.0
    overall_recall = total_tp_overall / total_annotations_overall if total_annotations_overall > 0 else 0.0
    overall_f1 = 0.0
    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

    # Calculate Overall IoU > threshold Image Ratio
    overall_iou_gt_threshold_ratio = 0.0
    if num_dataset_images > 0:
        overall_iou_gt_threshold_ratio = len(images_with_good_iou_overall) / num_dataset_images

    results = {
        'mAP': mAP,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1_Score': overall_f1,
        'Overall_IoU_GT_Threshold_Ratio': overall_iou_gt_threshold_ratio,
        'per_class_metrics': per_class_final_metrics
    }

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a RetinaNet model on the Knife dataset using csv_eval logic.')

    parser.add_argument('--dataset-root', default='./', help='Path to the root directory of the dataset.')
    parser.add_argument('--model-path', default='./knife_checkpoints/knife_retinanet_9.pt', help='Path to the trained model (.pt or .pth) file.')
    parser.add_argument('--set-name', help='Name of the test set to evaluate (e.g., test_knife, test_knife-1)', default='test_knife')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for mAP calculation.')
    parser.add_argument('--depth', help='Resnet depth used for training (18, 34, 50, 101, 152)', type=int, default=50)
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold for detections.')
    parser.add_argument('--max-detections', type=int, default=100, help='Max detections per image.')
    parser.add_argument('--no-cuda', help='Disable CUDA', action='store_true')
    parser.add_argument('--save-path', help='(Optional) Base path to save visualized prediction images (creates subdirs).', default=None) # Default None
    parser.add_argument('--cam-indices', help='(Optional) Comma-separated list of image indices (0-based) to generate Grad-CAM for.', default=None)

    args = parser.parse_args()

    use_gpu = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"Using device: {device}")

    if not args.dataset_root or not args.model_path:
        parser.error("Both --dataset-root and --model-path are required.")

    # --- Dataset ---
    transform = transforms.Compose([Normalizer(), Resizer()])
    try:
        dataset_eval = KnifeDataset(data_root=args.dataset_root, set_name=args.set_name, transform=transform)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        exit(1)

    if len(dataset_eval) == 0:
        print(f"Dataset '{args.set_name}' is empty or could not be loaded. Exiting.")
        exit(0)

    # Parse CAM indices if provided
    cam_indices_set = None
    if args.cam_indices:
        try:
            cam_indices_set = set(map(int, args.cam_indices.split(',')))
            print(f"Will generate Grad-CAM for image indices: {sorted(list(cam_indices_set))}")
        except ValueError:
            print("Error: Invalid format for --cam-indices. Please provide comma-separated integers (e.g., 0,5,12).")
            cam_indices_set = None # Disable if format is wrong
    elif args.save_path: # If saving is enabled but no indices given, warn but proceed without CAM
         print("Warning: --cam-indices not provided, but --save-path is set. CAM will not be generated.")
         cam_indices_set = set() # Empty set signifies CAM is possible but not requested for any index

    # --- Model ---
    print(f"Loading model from {args.model_path}...")
    try:
        num_classes = dataset_eval.num_classes()
        print(f"Creating model architecture: ResNet{args.depth} with {num_classes} classes.")
        model_creation_func = getattr(model, f'resnet{args.depth}')
        retinanet = model_creation_func(num_classes=num_classes, pretrained=False)
        state_dict = torch.load(args.model_path, map_location=device)
        retinanet.load_state_dict(state_dict)
        retinanet = retinanet.to(device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)
    except AttributeError:
         print(f"Error: Model function 'resnet{args.depth}' not found in retinanet.model.")
         exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Run Evaluation ---
    print("Starting evaluation...")
    # NOTE: evaluate now performs the main loop and metric calculations
    results = evaluate(dataset_eval, retinanet,
                       iou_threshold=args.iou_threshold,
                       score_threshold=args.score_threshold,
                       max_detections=args.max_detections,
                       device=device,
                       save_path=args.save_path,
                       cam_indices=cam_indices_set)

    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    print(f"Dataset: {args.set_name}")
    print(f"Model: {args.model_path}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"Score Threshold: {args.score_threshold}")
    print(f"Max Detections: {args.max_detections}")
    print(f"\n--- Overall Metrics ---")
    print(f"mAP@{args.iou_threshold:.2f}: {results['mAP']:.4f}")
    print(f"Overall Precision: {results['Precision']:.4f}")
    print(f"Overall Recall: {results['Recall']:.4f}")
    print(f"Overall F1 Score: {results['F1_Score']:.4f}")
    print(f"Overall IoU > {args.iou_threshold:.1f} Image Ratio: {results['Overall_IoU_GT_Threshold_Ratio']:.4f}")

    print("\n--- Per-class Metrics ---")
    iou_ratio_header = f"IoU>{args.iou_threshold:.1f}Ratio"
    print(f"{'Class':<20} {'AP':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {iou_ratio_header:<12} {'Annotations'}")
    print("-"*(75 + 12 + 1))
    for label, metrics in results['per_class_metrics'].items():
        class_name = dataset_eval.label_to_name(label)
        ap = metrics['AP']
        p = metrics['Precision']
        r = metrics['Recall']
        f1 = metrics['F1']
        iou_ratio = metrics['IoU_GT_Threshold_Ratio']
        num_ann = metrics['NumAnnotations']
        print(f"{class_name:<20} {ap:<10.4f} {p:<10.4f} {r:<10.4f} {f1:<10.4f} {iou_ratio:<12.4f} {int(num_ann)}")

    print("\nEvaluation finished.") 