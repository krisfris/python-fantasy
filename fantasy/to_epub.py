import os
import shutil
import subprocess


output_dir = 'data/epub'


def main():
    # Walk entire download folder
    for dirpath, dirs, files in os.walk('data/download'):
        # Group files by filename root (filename without extension)
        groups = dict()
        for filename in files:
            root, ext = os.path.splitext(filename)
            groups.setdefault(root, []).append((root, ext))

        # Iterate over each group
        for key, values in groups.items():
            # If there is an epub file in the group, copy it and be done
            epubs = [x for x in values if x[1] == '.epub']
            if epubs:
                filename = key + epubs[0][1]
                shutil.copyfile(os.path.join(dirpath, filename),
                                os.path.join(output_dir, filename))

            # Otherwise check for other accepted file types and convert it to epub
            else:
                other_exts = ['.mobi', '.lrf', '.rtf', '.lit', '.prc',
                              '.rar', '.zip', '.pdf', '.html', '.htm', '.opf']
                # Here I iterate over possible extensions instead of the values because order
                # matters: other_exts is sorted by most to least desirable for conversion
                other_ebooks = [(key, ext) for ext in other_exts if (key, ext) in values]
                if other_ebooks:
                    filename = key + other_ebooks[0][1]
                    subprocess.check_call(['ebook-convert', os.path.join(dirpath, filename),
                                           os.path.join(output_dir, key + '.epub')])


if __name__ == '__main__':
    main()
