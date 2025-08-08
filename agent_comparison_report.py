# Compare two experiment results

import pandas as pd
import plotly.graph_objects as go

# ---------------- CONFIG ---------------- #
run1_responses_path = 'results/gpt-4.1-mini-run1/all_experiment_results_20250720_065739.csv'  # set path 1 here
run2_responses_path = 'results/gemini-2.5-flash-run1/all_experiment_results_20250806_012956.csv'  # set path 2 here
context_path = 'data/indonesia_quarterly_context_2018_2025.csv'
output_html = 'post_treatment_expectation_comparison.html' # set output html name

# ---------------- FUNCTIONS ----------------
def load_and_merge(responses_path, context_path, run_label):
    responses = pd.read_csv(responses_path)
    context = pd.read_csv(context_path)

    df = responses.merge(context, on='quarter', how='left')
    df['run'] = run_label
    return df

def plot_post_treatment_comparison(df1, df2):
    # Combine and ensure correct types
    df_all = pd.concat([df1, df2], ignore_index=True)
    df_all['treatment_group'] = df_all['treatment_group'].astype(str)
    df_all['run'] = df_all['run'].astype(str)

    # Get all unique quarters in correct order
    def quarter_sort_key(q):
        import re
        m = re.match(r"(\d{4})Q(\d)", str(q))
        if m:
            return int(m.group(1)) * 10 + int(m.group(2))
        return 0

    all_quarters = sorted(df_all['quarter'].unique(), key=quarter_sort_key)

    # Load context directly to get correct inflation series
    context_df = pd.read_csv(context_path)
    context_df['quarter'] = context_df['quarter'].astype(str)
    context_df = context_df.set_index('quarter')
    inflation_map = context_df['current_inflation'].to_dict()
    inflation_values = [inflation_map.get(q, None) for q in all_quarters]

    figs = {}
    treatments = sorted(df_all['treatment_group'].unique())
    runs = ['gpt-4.1-mini', 'gemini-2.5-flash']

    for treatment in treatments:
        fig = go.Figure()
        for run in runs:
            means = []
            for q in all_quarters:
                mask = (
                    (df_all['treatment_group'] == treatment) &
                    (df_all['run'] == run) &
                    (df_all['quarter'] == q)
                )
                if mask.any():
                    means.append(df_all.loc[mask, 'post_treatment_expectation'].mean())
                else:
                    means.append(None)
            fig.add_trace(go.Scatter(
                x=all_quarters,
                y=means,
                mode='lines+markers',
                name=run
            ))
        # Add current actual inflation as a line
        fig.add_trace(go.Scatter(
            x=all_quarters,
            y=inflation_values,
            mode='lines+markers',
            name='Actual Inflation',
            line=dict(color='grey')
        ))
        fig.update_layout(
            title=f'Post Treatment Inflation Expectation: {treatment}',
            xaxis_title='Quarter',
            yaxis_title='Post Treatment Expectation',
            legend_title='Model',
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=all_quarters)
        figs[treatment] = fig
    return figs

# ---------------- MAIN ----------------
def main():
    df1 = load_and_merge(run1_responses_path, context_path, run_label='gpt-4.1-mini')
    df2 = load_and_merge(run2_responses_path, context_path, run_label='gemini-2.5-flash')

    figs = plot_post_treatment_comparison(df1, df2)
    from plotly.io import to_html

    all_figs_html = ""
    for treatment, fig in figs.items():
        all_figs_html += f"<h2>{treatment}</h2>" + to_html(fig, include_plotlyjs=False, full_html=False)

    full_html = f"""
    <html>
        <head>
            <title>Post Treatment Expectation Comparison</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Post Treatment Expectation Comparison by Treatment Group</h1>
            {all_figs_html}
        </body>
    </html>
    """

    with open(output_html, 'w') as f:
        f.write(full_html)

    print(f'All plots saved in: {output_html}')


if __name__ == '__main__':
    main()