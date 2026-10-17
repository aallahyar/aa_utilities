import pandas as pd
import pytest

from aa_utilities.helpers.plotly import build_sankey_data


@pytest.fixture
def link_df():
    return pd.DataFrame({'source': ['a', 'b', 'a', 'c'], 'target': ['x', 'y', 'y', 'x'], 'value': [1, 2, 3, 4]})


def test_inferred_nodes_list_sources_first_then_targets(link_df):
    node, link = build_sankey_data(link_df)

    assert node['label'] == ['a', 'b', 'c', 'x', 'y']
    assert link['source'] == [0, 1, 0, 2]
    assert link['target'] == [3, 4, 4, 3]
    assert link['value'] == [1, 2, 3, 4]


def test_inferred_nodes_ignore_the_link_index(link_df):
    link_df.index = [10, 11, 12, 13]
    assert build_sankey_data(link_df)[0]['label'] == ['a', 'b', 'c', 'x', 'y']


def test_inferred_nodes_with_numeric_ids():
    link_df = pd.DataFrame({'source': [3, 1], 'target': [1, 2], 'value': [5, 6]})
    node, link = build_sankey_data(link_df)

    assert node['label'] == ['3', '1', '2']  # 1 is a source, so it keeps its source position
    assert (link['source'], link['target']) == ([0, 1], [1, 2])


def test_explicit_node_table_sets_the_order(link_df):
    node_df = pd.DataFrame({'id': ['c', 'b', 'a', 'y', 'x'], 'label': list('CBAYX')})
    node, link = build_sankey_data(link_df, node_df)

    assert node['label'] == list('CBAYX')
    assert link['source'] == [2, 1, 2, 0]
    assert link['target'] == [4, 3, 3, 4]


def test_unknown_ids_and_duplicate_ids_raise(link_df):
    with pytest.raises(ValueError, match='unknown node id'):
        build_sankey_data(link_df, pd.DataFrame({'id': ['a', 'b'], 'label': ['A', 'B']}))
    with pytest.raises(ValueError, match='duplicates'):
        build_sankey_data(link_df, pd.DataFrame({'id': ['a', 'a'], 'label': ['A', 'A2']}))
