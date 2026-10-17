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


def test_link_value_column_is_required(link_df):
    with pytest.raises(ValueError, match='value'):
        build_sankey_data(link_df.drop(columns='value'))


def test_extra_columns_are_forwarded_under_their_own_names(link_df):
    link_df = link_df.assign(color=list('rgbk'), label=['l0', 'l1', 'l2', 'l3'], customdata=[10, 20, 30, 40])
    node_df = pd.DataFrame(
        {'id': ['a', 'b', 'c', 'x', 'y'], 'label': list('ABCXY'), 'color': list('12345'), 'hovertemplate': list('vwxyz')}
    )
    node, link = build_sankey_data(link_df, node_df)

    assert node == {'label': list('ABCXY'), 'color': list('12345'), 'hovertemplate': list('vwxyz')}
    assert set(link) == {'source', 'target', 'value', 'color', 'label', 'customdata'}
    assert link['label'] == ['l0', 'l1', 'l2', 'l3'] and link['customdata'] == [10, 20, 30, 40]


def test_id_source_and_target_are_translated_not_forwarded(link_df):
    node, link = build_sankey_data(link_df, pd.DataFrame({'id': list('abcxy'), 'label': list('ABCXY')}))
    assert 'id' not in node
    assert all(isinstance(position, int) for position in link['source'] + link['target'])


def test_node_positions_are_clipped_into_plotly_range(link_df):
    node_df = pd.DataFrame(
        {'id': list('abcxy'), 'label': list('ABCXY'), 'x': [0, 0.5, 1, 1.2, -0.1], 'y': [0, 0.25, 0.5, 0.75, 1]}
    )
    node, _ = build_sankey_data(link_df, node_df)

    assert node['x'] == [0.001, 0.5, 0.999, 0.999, 0.001]
    assert node['y'] == [0.001, 0.25, 0.5, 0.75, 0.999]


def test_link_index_does_not_matter(link_df):
    link_df.index = [10, 11, 12, 13]
    _, link = build_sankey_data(link_df.assign(color=list('rgbk')))
    assert link['source'] == [0, 1, 0, 2] and link['color'] == list('rgbk')


def test_tuple_ids_work():
    link_df = pd.DataFrame({'source': [('t0', 'g1')], 'target': [('t1', 'g2')], 'value': [3]})
    node_df = pd.DataFrame({'id': [('t0', 'g1'), ('t1', 'g2')], 'label': ['g1', 'g2']})
    node, link = build_sankey_data(link_df, node_df)

    assert node['label'] == ['g1', 'g2']
    assert (link['source'], link['target']) == ([0], [1])
