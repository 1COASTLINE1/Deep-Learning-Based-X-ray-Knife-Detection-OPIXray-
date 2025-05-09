import argparse
import collections
import os

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from dataloader import KnifeDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

# Import the evaluation function
from evaluate_knife import evaluate

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network on custom data.')

    parser.add_argument('--data_path', required=True, help='Path to the root directory containing train/ and test/ folders')
    parser.add_argument('--test_set', default='test_knife', help='Name of the test set directory (e.g., test_knife, test_knife-1)')
    parser.add_argument('--output_dir', default='./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--load_weights', default=None, help='(Optional) Path to a .pt file to load initial weights from')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--workers', help='Num workers for dataloader', type=int, default=8)
    parser.add_argument('--no_cuda', help='Disable CUDA', action='store_true')

    parser = parser.parse_args(args)

    use_gpu = not parser.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    transform_train = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
    transform_val = transforms.Compose([Normalizer(), Resizer()])

    try:
        dataset_train = KnifeDataset(data_root=parser.data_path, set_name='train', transform=transform_train)
        dataset_val = KnifeDataset(data_root=parser.data_path, set_name=parser.test_set, transform=transform_val)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure --data_path points to the root directory containing train/ and test/ subdirectories.")
        return
    except Exception as e:
         print(f"An unexpected error occurred during dataset loading: {e}")
         import traceback
         traceback.print_exc()
         return

    num_classes = dataset_train.num_classes()
    if num_classes == 0:
         print("Error: num_classes reported as 0. Check CLASS_MAP in dataloader.py")
         return
    print(f"Training with {num_classes} classes.")

    sampler_train = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler_train)

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler_val)

    print("Creating model...")
    model_creation_params = {'num_classes': num_classes}
    if parser.depth == 18:
        retinanet = model.resnet18(**model_creation_params, pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(**model_creation_params, pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(**model_creation_params, pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(**model_creation_params, pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(**model_creation_params, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if parser.load_weights:
         if os.path.exists(parser.load_weights):
             print(f"Loading weights from: {parser.load_weights}")
             try:
                 retinanet.load_state_dict(torch.load(parser.load_weights, map_location=device), strict=False)
             except Exception as e:
                 print(f"Error loading weights: {e}. Training from scratch or ImageNet backbone.")
         else:
             print(f"Warning: Weights file not found at {parser.load_weights}. Training from scratch or ImageNet backbone.")

    retinanet = retinanet.to(device)

    retinanet.training = True

    optimizer = optim.AdamW(retinanet.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    print('Num training images: {}'.format(len(dataset_train)))
    print('Num validation images: {}'.format(len(dataset_val)))

    os.makedirs(parser.output_dir, exist_ok=True)

    for epoch_num in range(parser.epochs):

        retinanet.train()
        if hasattr(retinanet, 'freeze_bn'):
            retinanet.freeze_bn()
        else:
             print("Warning: Model does not have freeze_bn method.")

        epoch_loss = []
        batch_iterator = iter(dataloader_train)

        for iter_num in range(len(dataloader_train)):
            try:
                data = next(batch_iterator)
                optimizer.zero_grad()

                images = data['img'].to(device).float()
                annotations = data['annot'].to(device)

                classification_loss, regression_loss = retinanet([images, annotations])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0) or not torch.isfinite(loss):
                    print(f"Warning: Invalid loss encountered (zero or inf/nan) at epoch {epoch_num}, iter {iter_num}. Skipping step.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                if iter_num % 50 == 0:
                     print(
                         f'Epoch: {epoch_num} | Iter: {iter_num}/{len(dataloader_train)} | Cls loss: {float(classification_loss):1.5f} | Reg loss: {float(regression_loss):1.5f} | Running loss: {np.mean(loss_hist):1.5f}'
                     )

                del classification_loss
                del regression_loss
                del loss

            except Exception as e:
                print(f"Error during training iteration {iter_num}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n--- Evaluation (Placeholder) ---")
        print(" Note: Evaluation logic needs to be adapted for KnifeDataset.")
        print("--- End Evaluation ---")

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(np.mean(epoch_loss))
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
             print(f"Learning rate changed to {new_lr}")

        checkpoint_path = os.path.join(parser.output_dir, f'knife_retinanet_{epoch_num}.pt')
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(retinanet.state_dict(), checkpoint_path)

        # --- Run Evaluation After Each Epoch ---
        print(f"\nRunning evaluation on '{parser.test_set}' set for epoch {epoch_num}...")
        try:
            # Pass the model currently in memory (on the correct device)
            # dataset_val is already loaded with the appropriate transforms
            eval_results = evaluate(dataset=dataset_val, model=retinanet, iou_threshold=0.5)
            print(f"--- Epoch {epoch_num} Evaluation Results ---")
            print(f"  mAP@0.5: {eval_results.get('mAP_0.5', 'N/A')}")
            print(f"  Recall (MAR_100): {eval_results.get('Recall (MAR_100)', 'N/A')}")
            print("-------------------------------------")
        except Exception as e:
            print(f"Error during evaluation for epoch {epoch_num}: {e}")
            import traceback
            traceback.print_exc()
        # --- End Evaluation ---


if __name__ == '__main__':
    main()
