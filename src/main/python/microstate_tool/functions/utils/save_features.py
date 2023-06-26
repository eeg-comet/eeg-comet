
import os.path


def save_features(df, filename, file_format, path):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, filename + file_format)
    if file_format == '.csv':
        df.to_csv(save_path, header=True, index=False)
    elif file_format == '.pkl':
        df.to_pickle(save_path)
    elif file_format == '.hdf':
        df.to_hdf(save_path, key='df', mode='w')
    elif file_format == '.json':
        df.to_json(save_path)
    else:
        raise ValueError("Failed to match file_format")

