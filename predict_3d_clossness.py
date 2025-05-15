import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors # Renamed to avoid conflict

from src.utils import load_pretrained_ernierna, prepare_input_for_ernierna # Assuming creatmat is in prepare_input
from src.downstream_heads.closeness_model import ClosenessModelAttnMapDense16 # Adjusted import

def rna_string_to_numerical_array(sequence_str):
    """
    Converts an RNA sequence string to a numerical array as expected by prepare_input_for_ernierna.
    A=5, U/T=6, C=7, G=4, N/other=3. Includes CLS (0) and EOS (2) tokens.
    """
    rna_map = {'A': 5, 'a': 5, 'U': 6, 'u': 6, 'T': 6, 't': 6, 'C': 7, 'c': 7, 'G': 4, 'g': 4}
    numerical_seq = [0] # CLS token
    for char_base in sequence_str:
        numerical_seq.append(rna_map.get(char_base, 3)) # Default to 3 (N/other) if not found
    numerical_seq.append(2) # EOS token
    return np.array(numerical_seq, dtype=np.int64)

def plot_closeness_map(matrix, title, save_path, cmap='viridis', clim_min=0, clim_max=1):
    """
    Plots a 2D matrix as a heatmap and saves it.
    """
    plt.style.use('default')
    plt.figure(figsize=(8, 6)) # Adjusted figure size for better visualization
    plt.imshow(matrix, cmap=cmap, origin='upper') # origin='upper' is common for matrices
    plt.clim(clim_min, clim_max)
    colorbar = plt.colorbar(ticks=np.linspace(clim_min, clim_max, 11))
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.title(title)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory
    # colorbar.remove() # Not needed if plt.close() is used

def load_sequences_from_file(file_path):
    """
    Loads sequences from a file.
    Handles FASTA (multi-line per sequence) or plain text (one sequence per line).
    Returns a list of (header, sequence) tuples.
    """
    sequences = []
    current_header = None
    current_sequence = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None and current_sequence:
                    sequences.append((current_header, "".join(current_sequence)))
                current_header = line[1:]
                current_sequence = []
            else:
                # If no header was found yet, and it's not a FASTA, treat as plain text
                if current_header is None and not sequences and len(line.split()) == 1: # Simple check
                    sequences.append((f"seq_{len(sequences)}", line)) # Assign a generic header
                elif current_header is not None: # Continue current FASTA sequence
                    current_sequence.append(line)
                elif not sequences: # First line, not FASTA, treat as plain
                     sequences.append((f"seq_{len(sequences)}", line))

        # Add the last sequence if any
        if current_header is not None and current_sequence:
            sequences.append((current_header, "".join(current_sequence)))
        elif not current_header and not sequences and current_sequence: # Case for single line plain text file
             sequences.append((f"seq_{len(sequences)}", "".join(current_sequence)))


    if not sequences:
        # Fallback for plain text file if FASTA parsing yielded nothing
        # but there was content (e.g. single sequence without header)
        # This part might need refinement based on exact plain text format expectations
        try:
            with open(file_path, 'r') as f: # Re-open
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        sequences.append((f"seq_{i}", line))
        except Exception as e:
            print(f"Could not parse file as FASTA or plain text: {e}")
            return []
            
    return sequences


def main(args):
    # Load ERNIE-RNA core model
    ernie_rna_core_model = load_pretrained_ernierna(
        args.ernie_rna_checkpoint_path,
        arg_overrides={"data": args.ernie_rna_dict_path}
    )
    ernie_rna_encoder = ernie_rna_core_model.encoder # Get the encoder part

    # Instantiate the downstream 3D Closeness model head
    # Assuming default mid_conv=8, drop_out=0.3 as in choose_model_dense16
    downstream_model = ClosenessModelAttnMapDense16(ernie_rna_encoder)

    # Load fine-tuned weights for the downstream model
    if not os.path.exists(args.finetuned_model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {args.finetuned_model_path}")
    
    # Important: The state_dict for the ClosenessModelAttnMapDense16 should only contain
    # weights for its own layers (conv1, relu, dropout, conv2, proj).
    # The ERNIE-RNA encoder part is already loaded with pre-trained weights.
    # If the saved .pt file contains the *entire* composite model (ERNIE-RNA + head),
    # then you'd load it differently, perhaps into a wrapper model.
    # For now, assuming args.finetuned_model_path contains *only* the head's weights,
    # or if it contains the full model, we need to be careful.
    
    # If the finetuned_model_path contains the state_dict of the *entire* ClosenessModelAttnMapDense16
    # (which includes the sentence_encoder attribute that is the ERNIE-RNA encoder)
    # then loading it directly is fine, but it means the ERNIE-RNA part might be overwritten
    # by fine-tuned ERNIE-RNA weights, which is usually intended.
    
    state_dict = torch.load(args.finetuned_model_path, map_location='cpu')
    
    # If the saved state_dict was from a DataParallel model
    if isinstance(downstream_model, torch.nn.DataParallel):
        downstream_model.module.load_state_dict(state_dict)
    elif any(key.startswith('module.') for key in state_dict.keys()):
        # Saved from DataParallel, but current model is not
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k # remove `module.`
            new_state_dict[name] = v
        downstream_model.load_state_dict(new_state_dict)
    else:
        downstream_model.load_state_dict(state_dict)

    downstream_model.to(args.device)
    downstream_model.eval()

    print("Models loaded successfully.")

    # Load sequences
    rna_sequences_with_headers = load_sequences_from_file(args.input_rna_file)
    if not rna_sequences_with_headers:
        print(f"No sequences found in {args.input_rna_file}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(args.plot_dir, exist_ok=True)

    for header, rna_seq_str in rna_sequences_with_headers:
        print(f"Processing: {header} (Length: {len(rna_seq_str)})")
        if len(rna_seq_str) > 1022 : # Max length ERNIE-RNA was trained on (1024 - 2 special tokens)
            print(f"Warning: Sequence {header} is longer than 1022nt and will be truncated.")
            rna_seq_str = rna_seq_str[:1022]

        # 1. Convert RNA string to numerical array for creatmat
        numerical_rna_array = rna_string_to_numerical_array(rna_seq_str)
        seq_len_no_special_tokens = len(rna_seq_str)

        # 2. Prepare 1D and 2D inputs for ERNIE-RNA
        # prepare_input_for_ernierna expects the numerical_rna_array (with CLS/EOS) and original seq_len
        one_d_input, two_d_input = prepare_input_for_ernierna(numerical_rna_array, seq_len_no_special_tokens)
        
        one_d_input = one_d_input.to(args.device)
        two_d_input = two_d_input.to(args.device)

        # 3. Perform inference
        with torch.no_grad():
            # The ClosenessModelAttnMapDense16's forward expects src_tokens and twod_tokens
            predicted_logits = downstream_model(one_d_input, two_d_input)
            predicted_probs = torch.sigmoid(predicted_logits).squeeze().cpu().numpy() # Squeeze batch and channel dim

        # 4. Save prediction
        # Sanitize header for filename
        safe_header = "".join(c if c.isalnum() else "_" for c in header)
        if not safe_header: safe_header = f"seq_{rna_sequences_with_headers.index((header, rna_seq_str))}"
        
        output_npy_path = os.path.join(args.output_dir, f"{safe_header}_closeness_pred.npy")
        np.save(output_npy_path, predicted_probs)
        print(f"Saved prediction for {header} to {output_npy_path}")

        # 5. Visualize if requested
        if args.visualize:
            plot_title = f"Predicted 3D Closeness - {header}\n(ERNIE-RNA attn-map dense16)"
            output_png_path = os.path.join(args.plot_dir, f"{safe_header}_closeness_pred.png")
            plot_closeness_map(predicted_probs, plot_title, output_png_path)
            print(f"Saved visualization for {header} to {output_png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict RNA 3D Closeness using ERNIE-RNA and a fine-tuned downstream head.")
    parser.add_argument("--input_rna_file", type=str, required=True, help="Path to RNA sequence file (FASTA or plain text list).")
    parser.add_argument("--output_dir", type=str, default="./results/ernie_rna_3d_clossness", help="Directory to save prediction .npy files.")
    
    parser.add_argument("--ernie_rna_checkpoint_path", type=str, default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', help="Path to the ERNIE-RNA pre-trained model checkpoint (.pt).")
    parser.add_argument("--ernie_rna_dict_path", type=str, default='./src/dict/', help="Path to the ERNIE-RNA dictionary directory.")
    parser.add_argument("--finetuned_model_path", type=str, default='./checkpoint/ERNIE-RNA_3d_clossness_checkpoint/ERNIE-RNA_3D_closeness_attnmap_dp16_finetuned.pt', help="Path to the fine-tuned 3D Closeness downstream model weights (.pt).")
    
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for computation (e.g., 'cuda:0', 'cpu').")
    
    parser.add_argument("--visualize", action='store_true', help="Generate and save plots of the predicted closeness maps.")
    parser.add_argument("--plot_dir", type=str, default="./results/ernie_rna_3d_clossness/closeness_plots/", help="Directory to save visualizations if --visualize is set.")

    # Potentially add arguments for mid_conv, drop_out if you want them configurable for ClosenessModelAttnMapDense16
    
    parsed_args = parser.parse_args()
    main(parsed_args)