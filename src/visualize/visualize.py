class Visualizer:
    def __init__(self, config_path):
        pass
    
    def print_labels_in_grid(labels):
        """
        Print the cluster labels in a grid format.
        :param labels: The list of labels.
        :param row_length: Number of labels to display per row.
        """
        print("Visualizing the document by topic clusters")
        row_length=30
        if len(labels) > 500:
            row_length=50
            
        if len(labels) <= row_length:
            print(labels)
            return
        
        for i in range(0, len(labels), row_length):
            # Slice the labels into rows and print them
            print(labels[i:i+row_length])