import os, sys, zipfile


def main():
    assert os.path.exists("dataset.zip")
    with zipfile.ZipFile("dataset.zip") as zf:
        zf.extractall()


if __name__ == "__main__":
    main()
