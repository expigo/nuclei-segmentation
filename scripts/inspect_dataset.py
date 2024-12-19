from nuclei_segmentation.data.dataset import NucleiDatasetInspector, DatasetSplit, MaskType

def main():
    # Initialize inspector
    inspector = NucleiDatasetInspector("./data/raw")
    
    # Visualize a sample
    inspector.visualize_sample(DatasetSplit.TRAIN, 0)
    
    # Get statistics for cell types
    stats = inspector.get_mask_statistics(DatasetSplit.TRAIN, MaskType.CELL_TYPES)
    print("\nCell Types Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Unique values: {sorted(stats['unique_values'])}")

if __name__ == "__main__":
    main()