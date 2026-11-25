import pytest

from aa_utilities.storage import Checkpoint, Container


def test_failed_save_keeps_the_previous_checkpoint(tmp_path):
    checkpoint = Checkpoint(path=tmp_path, verbose=False)
    checkpoint.save({'a': 1})

    with pytest.raises(Exception):
        checkpoint.save(lambda: None)  # not picklable

    assert checkpoint.load() == {'a': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['data.pkl.gz']  # no temp file left behind


def test_save_without_overwrite_refuses_existing_file(tmp_path):
    checkpoint = Checkpoint(path=tmp_path, verbose=False)
    checkpoint.save({'a': 1})

    with pytest.raises(FileExistsError):
        checkpoint.save({'a': 2}, overwrite=False)

    assert checkpoint.load() == {'a': 1}
    assert [p.name for p in tmp_path.iterdir()] == ['data.pkl.gz']


def test_save_and_load_roundtrip_overwrites(tmp_path):
    checkpoint = Checkpoint(path=tmp_path, verbose=False)
    checkpoint.save([1, 2])
    checkpoint.save([3])
    assert checkpoint.load() == [3]


def test_container_copy_keeps_underscore_and_non_string_keys():
    container = Container(a=1)
    container['_hidden'] = 2
    container[7] = 3

    copied = container.copy(deep=False)

    assert dict(copied) == {'a': 1, '_hidden': 2, 7: 3}
    assert dict(container.update(b=4)) == {'a': 1, '_hidden': 2, 7: 3, 'b': 4}
    assert dict(container) == {'a': 1, '_hidden': 2, 7: 3}  # update() does not mutate


def test_container_filter_and_drop_keep_non_string_keys():
    container = Container(a=1, b=2)
    container[7] = 3

    assert dict(container.drop(labels=['a'])) == {'b': 2, 7: 3}
    assert dict(container.filter(items=[7, 'a'])) == {7: 3, 'a': 1}


def test_overlapping_saves_do_not_share_a_temporary_file(tmp_path):
    checkpoint = Checkpoint(path=tmp_path, verbose=False)
    checkpoint.save('previous')

    class SavesWhilePickling:
        def __reduce__(self):
            checkpoint.save('inner')  # a second save to the same destination while this one is being written
            return (str, ('outer',))

    checkpoint.save(SavesWhilePickling())

    assert checkpoint.load() == 'outer'
    assert [p.name for p in tmp_path.iterdir()] == ['data.pkl.gz']
