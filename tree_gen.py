import os

def tree(dir_path='.', prefix='', exclude_folders=None, exclude_ext=None, f=None):
    if exclude_folders is None:
        exclude_folders = []
    if exclude_ext is None:
        exclude_ext = []

    files = sorted(os.listdir(dir_path))
    files = [x for x in files if not x.startswith('.') and x not in exclude_folders]

    for idx, filename in enumerate(files):
        path = os.path.join(dir_path, filename)
        connector = '├── ' if idx < len(files) - 1 else '└── '

        if os.path.isdir(path):
            print(prefix + connector + filename, file=f)
            extension = '│   ' if idx < len(files) - 1 else '    '
            tree(path, prefix + extension, exclude_folders, exclude_ext, f)
        else:
            if not any(filename.lower().endswith(ext) for ext in exclude_ext):
                print(prefix + connector + filename, file=f)

if __name__ == "__main__":
    with open('tree.txt', 'w', encoding='utf-8') as outfile:
        tree(
            '.',  # current folder
            exclude_folders=['.git', '__pycache__', 'venv', '.ipynb_checkpoints'],
            exclude_ext=['.jpg', '.jpeg', '.png'],
            f=outfile
        )