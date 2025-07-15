import pandas as pd
from inflation_exp import ExperimentManager, DatabaseManager, ResultsExporter
from datetime import datetime
import os
import json

# Set up database manager
db_path = 'data/inflation_study.db'
db_manager = DatabaseManager(db_type="sqlite", db_path=db_path)

# Load quarterly context data
quarter_df = pd.read_csv('data/indonesia_quarterly_context_2019_2025.csv')

# Directory to save results
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

all_results = []

for idx, row in quarter_df.iterrows():
    quarter = row['Quarter']
    print(f'Running experiment for quarter: {quarter}')
    # Prepare context dict for this quarter
    context = {
        'current_inflation': row['current_inflation'],
        'policy_rate': row['policy_rate'],
        'gdp_growth': row['gdp_growth'],
        'unemployment': row['unemployment'],
        'rupiah_exchange_rate': row['rupiah_exchange_rate'],
        'economic_outlook': row['economic_outlook'],
        'inflation_target': row['inflation_target'],
        'global_factors': row['global_factors'],
        'media_narrative': row['media_narrative'],
    }

    # Set up experiment manager
    exp_manager = ExperimentManager(db_manager)
    exp_manager.set_economic_context(context)

    # Run experiment for this quarter
    result = exp_manager.run_experiment(personas_per_group=30, quarter=quarter)
    all_results.append(result)

    # Export per quarter
    exporter = ResultsExporter(result, output_dir=RESULTS_DIR)

    base_name = f'experiment_results_{quarter}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exporter.export_to_json(base_name + '.json')
    exporter.export_to_csv(base_name + '.csv')

    report_name = f'report_{quarter}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exporter.generate_report(report_name + '.html')
    print(f'Exported results for {quarter}')

# Combine all results into one large file
combined = {
    'timestamp': datetime.now().isoformat(),
    'quarters': [r.get('results', []) for r in all_results],
    'all_results_flat': [item for r in all_results for item in r.get('results', [])]
}
combined_json = os.path.join(RESULTS_DIR, 'all_experiment_results.json')
with open(combined_json, 'w') as f:
    json.dump(combined, f, indent=2)

# Also export combined CSV
all_rows = [item for r in all_results for item in r.get('results', [])]
if all_rows:
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'all_experiment_results.csv'), index=False)

print(f'Combined results exported to {combined_json} and CSV.')
