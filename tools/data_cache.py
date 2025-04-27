import pickle
from tqdm import tqdm

from tools.raw_data_handle import get_paths_labels, get_mail_text

''' ========================= 数据缓存 ========================= '''


def _get_cache_path(data_cache_dir, name):
    return data_cache_dir / f"{name}.pkl"


def load_or_process_data(data_cache_dir, force_update=False):
    """智能加载或处理数据"""
    cache_files = {
        'path_list': _get_cache_path(data_cache_dir, 'path_list'),
        'label_list': _get_cache_path(data_cache_dir, 'label_list'),
        'content_list': _get_cache_path(data_cache_dir, 'content_list')
    }

    # 检查是否需要重新处理
    if not force_update and all(f.exists() for f in cache_files.values()):
        print("加载缓存数据...")
        with open(cache_files['path_list'], 'rb') as f:
            path_list = pickle.load(f)
        with open(cache_files['label_list'], 'rb') as f:
            label_list = pickle.load(f)
        with open(cache_files['content_list'], 'rb') as f:
            content_list = pickle.load(f)
    else:
        print("处理原始数据...")
        path_list, label_list = get_paths_labels()
        content_list = _process_content(path_list)

        # 保存缓存
        with open(cache_files['path_list'], 'wb') as f:
            pickle.dump(path_list, f)
        with open(cache_files['label_list'], 'wb') as f:
            pickle.dump(label_list, f)
        with open(cache_files['content_list'], 'wb') as f:
            pickle.dump(content_list, f)

    return path_list, label_list, content_list


def _process_content(path_list):
    """处理邮件内容（带进度条）"""
    content_list = []
    for path in tqdm(path_list, desc="Processing Emails"):
        try:
            content_list.append(get_mail_text(path))
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            content_list.append("")
    return content_list
