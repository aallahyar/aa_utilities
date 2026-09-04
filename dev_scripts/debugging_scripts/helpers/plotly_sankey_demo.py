import pandas as pd
import plotly.graph_objects as go

from aa_utilities.helpers.plotly import build_sankey_data

# ------- Data preparation -------
# Display text per node (can repeat — node_df['id'] below is the actual unique key).
label = ['Google Search', 'YouTube', 'AdMob', 'Google Play', 'Google Cloud', 'Other',
         'Ad Revenue', 'Revenue', 'Gross Profit', 'Cost of Revenues',
         'Operating Profit', 'Operating Expenses', 'TAC', 'Others',
         'Net Profit', 'Tax', 'Other', 'R&D', 'S&M', 'G&A']

# One color per node, same order as `label`. CSS names and hex RGB both work —
# mixed here on purpose ('#4682B4' == 'steelblue', '#FFD700' == 'gold', etc.).
node_color = ['#4682B4FF', 'steelblue', 'steelblue', '#FFD700', 'gold', 'gold',
              'steelblue', 'steelblue', '#008000', '#B22222',
              'green', 'firebrick', 'firebrick', 'firebrick',
              'green', 'firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick']

# Manual node position (0-1, left-to-right / top-to-bottom). Optional — omit x/y
# to let Plotly auto-arrange the nodes instead.
x = [0.12, 0.12, 0.12, 0.25, 0.30, 0.35, 0.35, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7,
     0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
y = [0.20, 0.42, 0.55, 0.70, 0.85, 0.95, 0.30, 0.40, 0.25, 0.70, 0.1, 0.40, 0.75, 0.90,
     0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
# Plotly rejects x/y values of exactly 0 or 1, so nudge them just inside the valid range.
clamp = lambda v: 0.001 if v == 0 else 0.999 if v == 1 else v
x, y = [clamp(v) for v in x], [clamp(v) for v in y]

# Flow endpoints — one entry per link, each referencing a node_df['id'] value (see below).
source = [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11]
target = [6, 6, 6, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# Flow size — controls the thickness of each link band.
value = [39.5, 7.1, 7.9, 6.9, 6.9, 0.88, 54.5, 37.9, 31.2, 17.1, 20.8,
         11.8, 19.3, 13.9, 2.3, 0.9, 10.3, 6.9, 3.6]

# One color per link, same order as source/target/value.
link_color = ['LightSkyBlue', 'LightSkyBlue', 'LightSkyBlue', 'goldenrod', 'goldenrod', 'goldenrod',
              'LightSkyBlue', 'lightgreen', 'PaleVioletRed', 'lightgreen', 'PaleVioletRed',
              'PaleVioletRed', 'PaleVioletRed', 'lightgreen', 'PaleVioletRed', 'PaleVioletRed',
              'PaleVioletRed', 'PaleVioletRed', 'PaleVioletRed']

# Optional short per-link text (distinct from hover), referenced via '%{label}' in link.hovertemplate.
# link_label = ['Ad Revenue', 'Ad Revenue', 'Ad Revenue', 'Revenue', 'Revenue', 'Revenue', 'Revenue',
#               'Gross Profit', 'Cost of Revenues', 'Operating Profit', 'Operating Expenses',
#               'TAC', 'Others', 'Net Profit', 'Tax', 'Other', 'R&D', 'S&M', 'G&A']

# `label` has duplicates here (e.g. two 'Other' nodes), so `id` must be a separate,
# guaranteed-unique key rather than the label itself.
node_df = pd.DataFrame({
    'id': range(len(label)),   # unique key referenced by link_df['source']/['target']
    'label': label,
    'color': node_color,
    'x': x,
    'y': y,
})
link_df = pd.DataFrame({
    'source': source,   # must match a value in node_df['id']
    'target': target,   # must match a value in node_df['id']
    'value': value,
    'color': link_color,
    # 'label': link_label,
})

print(node_df.iloc[:5, :])
print(link_df.iloc[:5, :])

# ------- producing the Sankey plot -------
node, link = build_sankey_data(node_df, link_df)
# link['label'] = link_label   # build_sankey_data doesn't pass this column through, so add it manually

fig = go.Figure(go.Sankey(
    textfont=dict(color='#000000', size=5),
    # orientation='v',   # vertical layout instead of the default horizontal
    # valueformat='.1f', valuesuffix='B',   # built-in numeric formatting, alternative to the manual "$%{value}B" below
    # arrangement='fixed',   # honor node_df['x']/['y'] exactly instead of letting Plotly nudge nodes to avoid overlap; options are: 'snap', 'perpendicular', 'freeform'
    node=dict(
        pad=35, thickness=20, line=dict(color='white', width=1),
        # hovertemplate='%{label}<extra></extra>',   # customize node hover text (default shows label + total flow)
        # customdata=node_df['label'],   # extra data available to node hovertemplate as '%{customdata}'
        # groups=[[17, 18, 19]],   # visually merge R&D/S&M/G&A (node_df row indices) into one collapsed node
        **node,
    ),
    link=dict(
        hovertemplate='%{source.label} \u2192 %{target.label}: $%{value}B<extra></extra>',
        # line=dict(color='white', width=0.5),   # border around each link band, mirroring node.line
        **link,
    ),
))

fig.update_layout(
    hovermode='x',
    title="<span style='font-size:36px;color:steelblue;'><b>Alphabet Q3 FY22 Income Statement</b></span>",
    font=dict(size=10, color='white'),
    paper_bgcolor='#F8F8FF',
)

fig.add_annotation(font=dict(color='steelblue', size=12), x=0.1, y=1.0, showarrow=False,
                    text='<b>Search advertising</b><br>$39.5B')
fig.add_annotation(font=dict(color='green', size=12), x=0.6, y=0.98, showarrow=False,
                    text='<b>Gross Profit</b><br>$69.1B (+6% Y/Y)')
fig.add_annotation(font=dict(color='maroon', size=12), x=0.98, y=1.05, showarrow=False,
                    text='<b>Net Profit</b><br>$13.9B (20% margin)')

fig.show() # open the interactive Sankey diagram in the default web browser

# Image export requires the Kaleido package: pip install --upgrade "kaleido>=1"
# fig.write_image("plotly_sankey_demo.pdf", width=1600, height=1000, scale=2)  # scale = 2x for high-DPI when an image is rendered
# fig.write_html("plotly_sankey_demo.html") # an interactive, self-contained file 
