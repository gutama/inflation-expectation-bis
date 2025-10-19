import pandas as pd
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

class ReasoningAnalyzer:
    """Analyzes pre and post reasoning from experiment results"""
    
    def __init__(self, csv_file: str):
        """Load and prepare data"""
        self.df = pd.read_csv(csv_file)
        self.treatment_groups = self.df['treatment_group'].unique()
        
    def extract_key_concepts(self, text: str) -> List[str]:
        """
        Extract key economic concepts from reasoning text
        
        Returns:
            List of identified concepts
        """
        concepts = {}
        
        # Define concept patterns
        concept_patterns = {
            'inflation_awareness': [r'inflation', r'price', r'increase.*price', r'cost of living'],
            'policy_awareness': [r'bank.*indonesia|BI|central bank', r'interest rate', r'monetary policy', r'policy'],
            'wage_concern': [r'wage', r'salary', r'income', r'earning'],
            'supply_chain': [r'supply', r'import', r'export', r'distribution', r'production'],
            'currency': [r'rupiah', r'exchange.*rate', r'currency', r'IDR'],
            'savings_behavior': [r'save', r'invest', r'deposit', r'spending'],
            'consumer_behavior': [r'spending', r'consumption', r'purchase', r'buy'],
            'employment': [r'job', r'employment', r'unemploy', r'business'],
            'global_factors': [r'global', r'world', r'international', r'usa|us|america', r'china'],
            'personal_experience': [r'i.*see|i.*notice|i.*feel|my.*experience', r'around.*me', r'local'],
        }
        
        text_lower = text.lower()
        identified = []
        
        for concept, patterns in concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    identified.append(concept)
                    break
                    
        return identified
    
    def compare_reasoning_by_treatment(self) -> Dict:
        """
        Compare reasoning patterns across treatment groups
        
        Returns:
            Dictionary with analysis by treatment group
        """
        results = {}
        
        for treatment in self.treatment_groups:
            treatment_data = self.df[self.df['treatment_group'] == treatment]
            
            pre_concepts = []
            post_concepts = []
            
            # Extract all concepts from pre and post reasoning
            for idx, row in treatment_data.iterrows():
                if pd.notna(row['pre_reasoning']):
                    pre_concepts.extend(self.extract_key_concepts(str(row['pre_reasoning'])))
                if pd.notna(row['post_reasoning']):
                    post_concepts.extend(self.extract_key_concepts(str(row['post_reasoning'])))
            
            # Count concept frequencies
            pre_counter = Counter(pre_concepts)
            post_counter = Counter(post_concepts)
            
            # Calculate shift in concepts
            all_concepts = set(pre_counter.keys()) | set(post_counter.keys())
            concept_shifts = {}
            
            for concept in all_concepts:
                pre_freq = pre_counter.get(concept, 0) / len(treatment_data) if len(treatment_data) > 0 else 0
                post_freq = post_counter.get(concept, 0) / len(treatment_data) if len(treatment_data) > 0 else 0
                concept_shifts[concept] = {
                    'pre': pre_freq,
                    'post': post_freq,
                    'change': post_freq - pre_freq
                }
            
            results[treatment] = {
                'sample_size': len(treatment_data),
                'pre_concept_counts': dict(pre_counter),
                'post_concept_counts': dict(post_counter),
                'concept_shifts': concept_shifts
            }
        
        return results
    
    def analyze_reasoning_length(self) -> Dict:
        """
        Analyze how treatment affects reasoning length and detail
        
        Returns:
            Dictionary with length analysis
        """
        results = {}
        
        for treatment in self.treatment_groups:
            treatment_data = self.df[self.df['treatment_group'] == treatment]
            
            pre_lengths = treatment_data['pre_reasoning'].fillna('').str.len()
            post_lengths = treatment_data['post_reasoning'].fillna('').str.len()
            
            results[treatment] = {
                'pre_avg_length': pre_lengths.mean(),
                'post_avg_length': post_lengths.mean(),
                'pre_std_length': pre_lengths.std(),
                'post_std_length': post_lengths.std(),
                'length_change': post_lengths.mean() - pre_lengths.mean()
            }
        
        return results
    
    def analyze_certainty_language(self) -> Dict:
        """
        Analyze changes in certainty/confidence language
        
        Returns:
            Dictionary with certainty analysis
        """
        certainty_patterns = {
            'high_certainty': [r'certain', r'sure', r'definitely', r'will', r'must', r'clearly'],
            'medium_certainty': [r'likely', r'probably', r'expect', r'should'],
            'low_certainty': [r'might', r'could', r'may', r'possibly', r'uncertain', r'think', r'believe'],
            'hedging': [r'but', r'however', r'although', r'while', r'on the other hand']
        }
        
        results = {}
        
        for treatment in self.treatment_groups:
            treatment_data = self.df[self.df['treatment_group'] == treatment]
            
            pre_certainty = {'high': [], 'medium': [], 'low': [], 'hedging': []}
            post_certainty = {'high': [], 'medium': [], 'low': [], 'hedging': []}
            
            for idx, row in treatment_data.iterrows():
                pre_text = str(row['pre_reasoning']).lower() if pd.notna(row['pre_reasoning']) else ''
                post_text = str(row['post_reasoning']).lower() if pd.notna(row['post_reasoning']) else ''
                
                # Check for certainty markers
                for level, patterns in certainty_patterns.items():
                    level_key = level.split('_')[0]  # 'high', 'medium', 'low'
                    
                    for pattern in patterns:
                        if re.search(pattern, pre_text):
                            if level not in pre_certainty:
                                pre_certainty[level_key] = []
                            pre_certainty[level_key].append(1)
                            break
                    
                    for pattern in patterns:
                        if re.search(pattern, post_text):
                            if level not in post_certainty:
                                post_certainty[level_key] = []
                            post_certainty[level_key].append(1)
                            break
            
            # Calculate percentages
            n = len(treatment_data)
            results[treatment] = {
                'pre': {k: len(v)/n if n > 0 else 0 for k, v in pre_certainty.items()},
                'post': {k: len(v)/n if n > 0 else 0 for k, v in post_certainty.items()}
            }
        
        return results
    
    def sample_reasoning_comparisons(self, treatment: str = None, n_samples: int = 5) -> List[Dict]:
        """
        Return sample pre/post reasoning pairs for inspection
        
        Args:
            treatment: Specific treatment to sample from (if None, randomly pick)
            n_samples: Number of samples to return
            
        Returns:
            List of sample reasoning pairs
        """
        if treatment:
            sample_df = self.df[self.df['treatment_group'] == treatment]
        else:
            sample_df = self.df
        
        samples = sample_df.sample(n=min(n_samples, len(sample_df)))
        
        result = []
        for idx, row in samples.iterrows():
            result.append({
                'persona_id': row['persona_id'],
                'treatment': row['treatment_group'],
                'pre_expectation': row['pre_treatment_expectation'],
                'post_expectation': row['post_treatment_expectation'],
                'expectation_change': row['expectation_change'],
                'pre_reasoning': row['pre_reasoning'][:200] + '...' if len(str(row['pre_reasoning'])) > 200 else row['pre_reasoning'],
                'post_reasoning': row['post_reasoning'][:200] + '...' if len(str(row['post_reasoning'])) > 200 else row['post_reasoning'],
                'confidence_change': row['post_confidence'] - row['pre_confidence']
            })
        
        return result
    
    def generate_reasoning_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive reasoning analysis report
        
        Args:
            output_file: Path to save HTML report
            
        Returns:
            HTML report content
        """
        # Run all analyses
        treatment_analysis = self.compare_reasoning_by_treatment()
        length_analysis = self.analyze_reasoning_length()
        certainty_analysis = self.analyze_certainty_language()
        
        html_content = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reasoning Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
                th { background-color: #34495e; color: white; }
                tr:nth-child(even) { background-color: #ecf0f1; }
                .section { margin: 30px 0; }
                .concept-shift { color: #27ae60; font-weight: bold; }
                .negative-shift { color: #e74c3c; }
            </style>
        </head>
        <body>
            <h1>Pre vs Post Treatment Reasoning Analysis</h1>
        '''
        
        # Section 1: Concept Shifts
        html_content += '<div class="section"><h2>1. Key Concept Shifts by Treatment</h2>'
        
        for treatment, analysis in treatment_analysis.items():
            html_content += f'<h3>{treatment.upper()} (n={analysis["sample_size"]})</h3>'
            html_content += '<table><tr><th>Concept</th><th>Pre %</th><th>Post %</th><th>Change</th></tr>'
            
            # Sort by absolute change
            sorted_shifts = sorted(
                analysis['concept_shifts'].items(),
                key=lambda x: abs(x[1]['change']),
                reverse=True
            )
            
            for concept, shifts in sorted_shifts[:10]:  # Top 10 changes
                change = shifts['change']
                change_class = 'concept-shift' if change > 0 else 'negative-shift' if change < 0 else ''
                html_content += f'''
                <tr>
                    <td>{concept}</td>
                    <td>{shifts["pre"]*100:.1f}%</td>
                    <td>{shifts["post"]*100:.1f}%</td>
                    <td class="{change_class}">{change*100:+.1f}%</td>
                </tr>
                '''
            
            html_content += '</table>'
        
        html_content += '</div>'
        
        # Section 2: Reasoning Length
        html_content += '<div class="section"><h2>2. Reasoning Length Analysis</h2>'
        html_content += '<table><tr><th>Treatment</th><th>Pre Avg Length</th><th>Post Avg Length</th><th>Change</th></tr>'
        
        for treatment, lengths in length_analysis.items():
            change = lengths['length_change']
            change_class = 'concept-shift' if change > 0 else 'negative-shift' if change < 0 else ''
            html_content += f'''
            <tr>
                <td>{treatment}</td>
                <td>{lengths["pre_avg_length"]:.0f} chars</td>
                <td>{lengths["post_avg_length"]:.0f} chars</td>
                <td class="{change_class}">{change:+.0f} chars</td>
            </tr>
            '''
        
        html_content += '</table></div>'
        
        # Section 3: Certainty Language
        html_content += '<div class="section"><h2>3. Certainty Language Patterns</h2>'
        
        for treatment, certainty_data in certainty_analysis.items():
            html_content += f'<h3>{treatment.upper()}</h3>'
            html_content += '<table><tr><th>Certainty Level</th><th>Pre %</th><th>Post %</th><th>Change</th></tr>'
            
            for level in ['high', 'medium', 'low', 'hedging']:
                pre_pct = certainty_data['pre'].get(level, 0) * 100
                post_pct = certainty_data['post'].get(level, 0) * 100
                change = post_pct - pre_pct
                change_class = 'concept-shift' if change > 0 else 'negative-shift' if change < 0 else ''
                
                html_content += f'''
                <tr>
                    <td>{level.capitalize()}</td>
                    <td>{pre_pct:.1f}%</td>
                    <td>{post_pct:.1f}%</td>
                    <td class="{change_class}">{change:+.1f}%</td>
                </tr>
                '''
            
            html_content += '</table>'
        
        html_content += '</div>'
        
        html_content += '''
        </body>
        </html>
        '''
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Report saved to {output_file}")
        
        return html_content


# Usage example
if __name__ == "__main__":
    # Load and analyze
    analyzer = ReasoningAnalyzer('results/gpt-4.1-mini-exclusive-treatment-run2/all_experiment_results_20250821_054934.csv')
    
    # Get treatment analysis
    print("=" * 60)
    print("CONCEPT SHIFT ANALYSIS BY TREATMENT")
    print("=" * 60)
    
    treatment_results = analyzer.compare_reasoning_by_treatment()
    for treatment, analysis in treatment_results.items():
        print(f"\n{treatment.upper()} (n={analysis['sample_size']})")
        print("-" * 60)
        
        # Top shifts
        sorted_shifts = sorted(
            analysis['concept_shifts'].items(),
            key=lambda x: abs(x[1]['change']),
            reverse=True
        )
        
        for concept, shifts in sorted_shifts[:5]:
            print(f"  {concept:20} | Pre: {shifts['pre']*100:5.1f}% | Post: {shifts['post']*100:5.1f}% | Change: {shifts['change']*100:+6.1f}%")
    
    # Get length analysis
    print("\n" + "=" * 60)
    print("REASONING LENGTH ANALYSIS")
    print("=" * 60)
    
    length_results = analyzer.analyze_reasoning_length()
    for treatment, lengths in length_results.items():
        print(f"\n{treatment.upper()}")
        print(f"  Pre avg:  {lengths['pre_avg_length']:.0f} characters")
        print(f"  Post avg: {lengths['post_avg_length']:.0f} characters")
        print(f"  Change:   {lengths['length_change']:+.0f} characters")
    
    # Get certainty analysis
    print("\n" + "=" * 60)
    print("CERTAINTY LANGUAGE ANALYSIS")
    print("=" * 60)
    
    certainty_results = analyzer.analyze_certainty_language()
    for treatment, certainty in certainty_results.items():
        print(f"\n{treatment.upper()}")
        print("  High Certainty:   Pre {:.1f}% → Post {:.1f}%".format(
            certainty['pre'].get('high', 0)*100, certainty['post'].get('high', 0)*100))
        print("  Medium Certainty: Pre {:.1f}% → Post {:.1f}%".format(
            certainty['pre'].get('medium', 0)*100, certainty['post'].get('medium', 0)*100))
        print("  Low Certainty:    Pre {:.1f}% → Post {:.1f}%".format(
            certainty['pre'].get('low', 0)*100, certainty['post'].get('low', 0)*100))
        print("  Hedging Language: Pre {:.1f}% → Post {:.1f}%".format(
            certainty['pre'].get('hedging', 0)*100, certainty['post'].get('hedging', 0)*100))
    
    # Sample reasoning pairs
    print("\n" + "=" * 60)
    print("SAMPLE REASONING COMPARISONS (Control Group)")
    print("=" * 60)
    
    samples = analyzer.sample_reasoning_comparisons('control', n_samples=3)
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i} (Persona: {sample['persona_id']})")
        print(f"  Change: {sample['pre_expectation']:.1f}% → {sample['post_expectation']:.1f}% ({sample['expectation_change']:+.1f}%)")
        print(f"  Confidence change: {sample['confidence_change']:+d}")
        print(f"  PRE:  {sample['pre_reasoning']}")
        print(f"  POST: {sample['post_reasoning']}")
    
    # Generate HTML report
    analyzer.generate_reasoning_report('reasoning_analysis_report.html')
    print("\n✓ Comprehensive report saved to reasoning_analysis_report.html")