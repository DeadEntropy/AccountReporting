from bkanalysis.data_manager import DataManager


def main():
    dm = DataManager()
    dm.load_data_from_disk()
    return dm.data


if __name__ == "__main__":
    main()
