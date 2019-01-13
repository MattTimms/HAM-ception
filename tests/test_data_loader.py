from __future__ import print_function
import os
import shutil

import pytest

from common.data_loader import import_ham_dataset, HAMDatasetException


class TestDataLoader:
    HAM_SIZE = 10015  # ttl size of HAM10000 dataset
    CUR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    DATA_DIR = os.path.join(CUR_DIR, '..', 'dataset/skin-cancer-mnist-ham10000/')

    @pytest.fixture(scope='function', autouse=True)
    def clean_test_dir(self):
        test_yaml = os.path.join(self.DATA_DIR, 'session.yaml')
        if os.path.exists(test_yaml):
            os.remove(test_yaml)
        yield
        if os.path.exists(test_yaml):
            os.remove(test_yaml)

    def test_training(self):
        dataset = import_ham_dataset(dataset_root=self.DATA_DIR, model_path=self.DATA_DIR, training=True)
        assert len(dataset) == (self.HAM_SIZE - dataset.num_test_imgs)
        assert os.path.exists(dataset.path_yaml)

    def test_testing(self):
        _ = import_ham_dataset(dataset_root=self.DATA_DIR, model_path=self.DATA_DIR, training=True)
        dataset = import_ham_dataset(dataset_root=self.DATA_DIR, model_path=self.DATA_DIR, training=False)
        assert len(dataset) == dataset.num_test_imgs

    def test_testing_wo_yaml(self):
        with pytest.raises(HAMDatasetException):
            _ = import_ham_dataset(dataset_root=self.DATA_DIR, model_path=self.DATA_DIR, training=False)
