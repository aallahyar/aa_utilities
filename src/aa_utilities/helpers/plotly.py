import pandas as pd

def build_sankey_data(node_df: pd.DataFrame, link_df: pd.DataFrame) -> tuple[dict, dict]:
    """Turn a node table and a link table into the `node`/`link` dicts that
    `go.Sankey(node=..., link=...)` expects, so callers can work with
    human-readable, arbitrary node ids instead of manually managing the
    positional (0-based) indices Plotly requires.

    Args:
        node_df: One row per node, with columns:
            - id (required): arbitrary, unique, hashable key (int, str, ...)
              used to reference this node from `link_df['source']`/`['target']`.
              Row order in `node_df` determines the 0-based position each id
              is mapped to (i.e. node order as passed to `go.Sankey`).
            - label (required): text displayed on/for the node. Unlike `id`,
              this does not need to be unique (e.g. two nodes can both be
              labeled "Other").
            - color (optional): per-node fill color (name or hex string).
              Omit the column entirely to let Plotly pick colors.
            - x, y (optional, both required together): manual node position
              in the `[0, 1]` plot area. Plotly rejects values of exactly 0
              or 1, so nudge those to e.g. 0.001/0.999 before calling this.
              Omit both columns to let Plotly auto-arrange nodes.
        link_df: One row per flow, with columns:
            - source, target (required): values that must exist in
              `node_df['id']`; define which nodes each flow connects.
            - value (required): flow size; controls the link's band thickness.
            - color (optional): per-link color (name or hex string).

    Returns:
        (node, link) dict pair suitable for `go.Sankey(node=node, link=link)`
        (merge in extra display kwargs like `pad`/`thickness`/`hovertemplate`
        as needed).

    Raises:
        ValueError: if `node_df['id']` has duplicates, or `link_df['source']`/
            `['target']` references an id not present in `node_df['id']`.
    """

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

    node = dict(label=node_df['label'].tolist())
    if 'color' in node_df:
        node['color'] = node_df['color'].tolist()
    if {'x', 'y'}.issubset(node_df.columns):
        node['x'] = node_df['x'].tolist()
        node['y'] = node_df['y'].tolist()

    link = dict(
        source=source_pos.astype(int).tolist(),
        target=target_pos.astype(int).tolist(),
        value=link_df['value'].tolist(),
    )
    if 'color' in link_df:
        link['color'] = link_df['color'].tolist()

    return node, link
