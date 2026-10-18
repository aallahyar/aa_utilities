import pandas as pd

# Plotly rejects node positions of exactly 0 or 1
_POSITION_MIN, _POSITION_MAX = 0.001, 0.999


def _as_lists(df: pd.DataFrame, exclude=()) -> dict:
    return {column: df[column].tolist() for column in df.columns if column not in exclude}


def build_sankey_data(link_df: pd.DataFrame, node_df: pd.DataFrame = None) -> tuple[dict, dict]:
    """Turn a link table and a node table into the `link`/`node` dicts that
    `go.Sankey(node=..., link=...)` expects, so callers can work with
    human-readable, arbitrary node ids instead of manually managing the
    positional (0-based) indices Plotly requires.

    Only the columns that need translating are treated specially (`id`, `source`,
    `target`). Every other column is forwarded, as a list, under its own name, so any
    per-node/per-link Plotly attribute works (`label`, `color`, `x`, `y`, `customdata`,
    `hovertemplate`, ...). Plotly rejects names it does not know, so select the columns
    you want before calling.

    Args:
        link_df: One row per flow, with columns:
            - source, target (required): values that must exist in
              `node_df['id']`; define which nodes each flow connects.
            - value (required): flow size; controls the link's band thickness.
            - any other per-link attribute, e.g. `color` (name or hex string) or `label`.
        node_df: One row per node, with columns:
            - id (required): arbitrary, unique, hashable key (int, str, tuple, ...)
              used to reference this node from `link_df['source']`/`['target']`.
              Row order in `node_df` determines the 0-based position each id
              is mapped to (i.e. node order as passed to `go.Sankey`).
              It is not forwarded to Plotly.
            - label: text displayed on/for the node. Unlike `id`,
              this does not need to be unique (e.g. two nodes can both be
              labeled "Other").
            - x, y: manual node position in the `[0, 1]` plot area. Values outside
              (0, 1) are clipped to 0.001/0.999, because Plotly rejects exactly 0 or 1.
            - any other per-node attribute, e.g. `color`.
            If `node_df` is omitted, its rows are inferred from `link_df['source']`/`['target']`:
            sources first, then the remaining targets, each in order of first appearance,
            with `label` defaulting to `str(id)`.

    Returns:
        (link, node) dict pair suitable for `go.Sankey(link=link, node=node)`
        (merge in extra display kwargs like `pad`/`thickness`/`hovertemplate`
        as needed).

    Raises:
        ValueError: if `link_df` has no `value` column, if `node_df['id']` has duplicates, or
            `link_df['source']`/`['target']` references an id not present in `node_df['id']`.
    """

    if 'value' not in link_df:
        raise ValueError("link_df needs a 'value' column")

    if node_df is None:
        ids = pd.concat([link_df['source'], link_df['target']]).drop_duplicates(ignore_index=True)
        node_df = pd.DataFrame({'id': ids, 'label': ids.astype(str)})

    if node_df['id'].duplicated().any():
        dupes = node_df.loc[node_df['id'].duplicated(), 'id'].tolist()
        raise ValueError(f"node_df['id'] has duplicates: {dupes}")

    # Map each node id to its row position -- this is the 0-based index
    # go.Sankey's `link.source`/`link.target` actually require.
    id_to_pos = pd.Series(range(len(node_df)), index=node_df['id'])

    source_pos = link_df['source'].map(id_to_pos)
    target_pos = link_df['target'].map(id_to_pos)
    # Unmapped ids come back as NaN from `.map`; collect them for a clear error
    # instead of silently emitting NaN indices into `link['source']/['target']`.
    unknown = pd.concat([
        link_df.loc[source_pos.isna(), 'source'],
        link_df.loc[target_pos.isna(), 'target'],
    ]).unique()
    if len(unknown) > 0:
        raise ValueError(f"link_df references unknown node id(s): {list(unknown)}")

    node = _as_lists(node_df, exclude=('id',))
    for axis in ('x', 'y'):
        if axis in node_df:
            node[axis] = node_df[axis].clip(_POSITION_MIN, _POSITION_MAX).tolist()

    link = _as_lists(link_df, exclude=('source', 'target')) | dict(
        source=source_pos.astype(int).tolist(),
        target=target_pos.astype(int).tolist(),
    )

    return link, node
