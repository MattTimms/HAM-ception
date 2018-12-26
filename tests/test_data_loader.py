from __future__ import print_function
import os
import shutil

import pytest

from common.data_loader import import_ham_dataset, HAMDatasetException


class TestDataLoader:
    HAM_SIZE = 10015  # ttl size of HAM10000 dataset
    CUR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    DATA_DIR = os.path.join(CUR_DIR, '..', 'dataset/skin-cancer-mnist-ham10000/')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    @pytest.fixture(scope='function', autouse=True)
    def clean_test_dir(self):
        yield
        if os.path.exists(self.TEST_DIR) and os.path.isdir(self.TEST_DIR):
            shutil.rmtree(self.TEST_DIR)

    def test_dataset_training(self):
        dataset = import_ham_dataset(dataset_root=self.DATA_DIR, training=True)
        assert len(dataset) == (TestDataLoader.HAM_SIZE - dataset.num_test_imgs)
        assert os.path.exists(dataset.dir_test)
        assert os.path.isdir(dataset.dir_test)
        assert len(os.listdir(dataset.dir_test)) == dataset.num_test_imgs

    def test_dataset_testing(self):
        _ = import_ham_dataset(dataset_root=self.DATA_DIR, training=True)  # setup a testing folder
        dataset = import_ham_dataset(dataset_root=TestDataLoader.DATA_DIR, training=False)
        assert len(dataset) == dataset.num_test_imgs

    def test_dataset_testing_wo_folder(self):
        try:
            _ = import_ham_dataset(dataset_root=self.DATA_DIR, training=False)
        except Exception as err:
            assert isinstance(err, HAMDatasetException)
