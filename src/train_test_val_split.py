import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_data_pts = 79

    n_test = int(n_data_pts * 0.2)
    n_val = int(n_data_pts * 0.2)
    n_train = n_data_pts - n_test - n_val
    
    indices = np.arange(n_data_pts)
    rng.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    print(f"Train indices: {train_indices}")
    print(f"Validation indices: {val_indices}")
    print(f"Test indices: {test_indices}")
    
    np.savetxt("../data/train_indices.txt", train_indices, fmt="%d")
    np.savetxt("../data/val_indices.txt", val_indices, fmt="%d")
    np.savetxt("../data/test_indices.txt", test_indices, fmt="%d")


