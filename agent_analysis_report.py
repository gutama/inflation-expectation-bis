# Generate a report for one experiment

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------- CONFIG ---------------- #
responses_path = 'results/gemini-2.5-flash-run1/all_experiment_results_20250806_012956.csv' # set path to response
context_path = 'data/indonesia_quarterly_context_2018_2025.csv'
sk_survey_path = 'data/survei_konsumen_2018_2020.csv'
output_html = 'report_gemini-2.5-flash.html' # set report name

# ---------------- FUNCTIONS ----------------
def load_data(responses_path, context_path, sk_survey_path):
    responses = pd.read_csv(responses_path)
    context = pd.read_csv(context_path)
    sk_survey = pd.read_csv(sk_survey_path)
    return responses, context, sk_survey

def merge_data(responses,  context):
    return responses.merge(context, on='quarter', how='left')

def plot_expectation_change(df):
    grouped = df.groupby(['quarter', 'treatment_group'])['expectation_change'].mean().reset_index()
    pivot_df = grouped.pivot(index='quarter', columns='treatment_group', values='expectation_change').reset_index()
    fig = go.Figure()
    for col in pivot_df.columns[1:]:
        fig.add_trace(go.Scatter(x=pivot_df['quarter'], y=pivot_df[col], mode='lines+markers', name=col))
    fig.update_layout(title='Mean Expectation Change per Quarter by Treatment Group',
                      xaxis_title='Quarter', yaxis_title='Mean Expectation Change',
                      xaxis_tickangle=-45, template='plotly_white')
    return fig

def plot_post_treatment(df, context):
    grouped = df.groupby(['quarter', 'treatment_group'])['post_treatment_expectation'].mean().reset_index()
    pivot_df = grouped.pivot(index='quarter', columns='treatment_group', values='post_treatment_expectation').reset_index()
    fig = go.Figure()
    for col in pivot_df.columns[1:]:
        fig.add_trace(go.Scatter(x=pivot_df['quarter'], y=pivot_df[col], mode='lines+markers', name=col))
    fig.add_trace(go.Scatter(x=context['quarter'], y=context['current_inflation'],
                             mode='lines+markers', name='Actual Inflation', line=dict(color='grey')))
    fig.update_layout(title='Post Treatment Inflation Expectation by Treatment Group',
                      xaxis_title='Quarter', yaxis_title='Inflation Expectation',
                      xaxis_tickangle=-45, template='plotly_white')
    return fig

def plot_treatment_effects(df):
    treatment_means = df.groupby('treatment_group')['expectation_change'].agg(['mean', 'std']).reset_index()
    treatment_means['error'] = treatment_means['std'] * 1.96
    fig = px.bar(treatment_means, x='treatment_group', y='mean', error_y='error',
                 text=treatment_means['mean'].round(2),
                 title='Average Treatment Effect (ATE) in Inflation Expectations per Treatment Group',
                 labels={'mean': 'Average Treatment Effect', 'treatment_group': 'Treatment Group'},
                 color='treatment_group', color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, template='plotly_white', yaxis_title='ATE')
    return fig

def plot_sk_comparison(df, sk_survey):
    df_18_20 = sk_survey.merge(df, how='left', on='quarter')
    grouped = df_18_20.groupby(['quarter', 'treatment_group'])['post_treatment_expectation'].mean().reset_index()
    pivot_df = grouped.pivot(index='quarter', columns='treatment_group', values='post_treatment_expectation').reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col in pivot_df.columns[1:]:
        fig.add_trace(
            go.Scatter(
                x=pivot_df['quarter'],
                y=pivot_df[col],
                mode='lines+markers',
                name=col
            ),
            secondary_y=False
        )

    fig.add_trace(
        go.Scatter(
            x=sk_survey['quarter'],
            y=sk_survey['inflation_expectation'],
            mode='lines+markers',
            name='Survei Konsumen',
            line=dict(color='slategrey'),
            marker=dict(symbol='circle')
        ),
        secondary_y=False
    )

    fig.update_layout(
        title='Post Treatment Inflation Expectation vs Survei Konsumen Bank Indonesia',
        xaxis_title='Quarter',
        yaxis_title='Treatment Group Expectation',
        legend_title='Treatment Group',
        xaxis_tickangle=-45,
        template='plotly_white'
    )

    # # for secondary axis
    # fig.update_yaxes(title_text='Survei Konsumen Inflation Expectation', secondary_y=True)

    return fig

def plot_post_treatment_vs_shifted_inflation(df, context):
    import pandas as pd
    import plotly.graph_objects as go

    # Group and pivot
    grouped = df.groupby(['quarter', 'treatment_group'])['post_treatment_expectation'].mean().reset_index()
    pivot_df = grouped.pivot(index='quarter', columns='treatment_group', values='post_treatment_expectation')
    pivot_df_reset = pivot_df.reset_index()

    quarters = pd.PeriodIndex(context['quarter'], freq='Q')
    shifted_quarters = quarters - 4

    # Filter out anything earlier than the first quarter in your main data
    valid_mask = shifted_quarters >= pd.Period(min(pivot_df_reset['quarter']), freq='Q')
    shifted_quarters_str = shifted_quarters[valid_mask].astype(str)
    shifted_inflation = pd.Series(context['current_inflation'])[valid_mask].reset_index(drop=True)

    fig = go.Figure()

    # Treatment group lines
    for treatment in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df_reset['quarter'],
            y=pivot_df_reset[treatment],
            mode='lines+markers',
            name=treatment
        ))

    # Shifted actual inflation, clipped
    fig.add_trace(go.Scatter(
        x=shifted_quarters_str,
        y=shifted_inflation,
        mode='lines+markers',
        name='Actual Inflation (1Y Ahead)',
        line=dict(color='grey'),
        marker=dict(symbol='circle')
    ))

    fig.update_layout(
        title="Post Treatment Mean Inflation Expectation per Quarter by Treatment Group vs Actual Inflation (1 Year Ahead), gemini-2.5-flash",
        xaxis_title="Quarter",
        yaxis_title="Inflation Expectation",
        xaxis_tickangle=-45,
        template="plotly_white",
        legend_title="Treatment Group"
    )

    return fig

# ---------------- MAIN ----------------
def main():
    responses, context, sk_survey = load_data(responses_path, context_path, sk_survey_path)
    df = merge_data(responses, context)

    fig1 = plot_expectation_change(df)
    fig2 = plot_post_treatment(df, context)
    fig3 = plot_treatment_effects(df)
    fig4 = plot_sk_comparison(df, sk_survey)
    fig5 = plot_post_treatment_vs_shifted_inflation(df, context)

    # Convert each figure to HTML
    fig1_html = pio.to_html(fig1, include_plotlyjs='cdn', full_html=False)
    fig2_html = pio.to_html(fig2, include_plotlyjs='cdn', full_html=False)
    fig3_html = pio.to_html(fig3, include_plotlyjs='cdn', full_html=False)
    fig4_html = pio.to_html(fig4, include_plotlyjs='cdn', full_html=False)
    fig5_html = pio.to_html(fig5, include_plotlyjs='cdn', full_html=False)

    # Add text between figures
    section1 = "<h2>Expectation Change</h2><p>This plot shows the mean change in inflation expectations by treatment group over time.</p>"
    section2 = "<h2>Post-Treatment Expectation</h2><p>This plot compares post-treatment expectations to actual inflation.</p>"
    section3 = "<h2>Treatment Effects</h2><p>This plot summarizes the average treatment effect (ATE) for each group.</p>"
    section4 = "<h2>Comparison with Survei Konsumen</h2><p>This plot compares the agent survey results with Bank Indonesia's household consumer survey</p>"
    section5 = "<h2>Post-Treatment vs Actual Inflation (1 Year Ahead)</h2><p>This plot compares post-treatment expectations to actual inflation shifted 1 year ahead.</p>"

    # Combine into a single HTML file
    full_html = f"""
    <html>
        <head><title>Inflation Report</title></head>
        <body>
            <h1>Inflation Expectations Analysis Report</h1>
            <p><strong>Model:</strong> {output_html}</p>
            {section1}
            {fig1_html}

            {section2}
            {fig2_html}

            {section3}
            {fig3_html}

            {section4}
            {fig4_html}

            {section5}
            {fig5_html}
        </body>
        </html>
    """

    with open(output_html, 'w') as f:
        f.write(full_html)

    print(f'Report generated: {output_html}')

if __name__ == '__main__':
    main()