"""Latest-iteration discovery should only consider iteration directories."""
from utils import system_utils as target
import pytest

@pytest.mark.parametrize('noise', ['.DS_Store', 'README', 'notes_999999', 'iteration_backup'])
def test_unrelated_files_do_not_break_discovery(tmp_path, noise):
    (tmp_path / 'iteration_7').mkdir()
    (tmp_path / 'iteration_30').mkdir()
    (tmp_path / noise).write_text('unrelated')
    assert target.searchForMaxIteration(tmp_path) == 30

def test_named_file_is_not_a_checkpoint_directory(tmp_path):
    (tmp_path / 'iteration_5').mkdir()
    (tmp_path / 'iteration_999').write_text('not a directory')
    assert target.searchForMaxIteration(tmp_path) == 5

def test_numeric_not_lexicographic_order(tmp_path):
    for name in ('iteration_9', 'iteration_10', 'iteration_0'):
        (tmp_path / name).mkdir()
    assert target.searchForMaxIteration(tmp_path) == 10

def test_empty_folder_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match='No iteration directories'):
        target.searchForMaxIteration(tmp_path)
