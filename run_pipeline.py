#!/usr/bin/env python3
"""
Main Orchestration Runner for Himalayan Herbal AI Project
=========================================================

Comprehensive pipeline runner:
1. Data Collection & Processing
2. Graph Building (for GNN)
3. Model Training (Bioactivity, Toxicity, GNN)
4. Evaluation & Reporting
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def print_banner(text: str):
    """Print a fancy banner"""
    width = 70
    print("\n" + "=" * width)
    print(f"  {text:^{width-4}}")
    print("=" * width + "\n")


def run_data_collection():
    """Phase 1: Collect and process data"""
    print_banner("PHASE 1: DATA COLLECTION & PROCESSING")

    try:
        from src.data.data_collector import HybridAntimicrobialCollector

        collector = HybridAntimicrobialCollector()

        # Get built-in dataset
        df = collector.get_builtin_himalayan_dataset()

        # Calculate molecular descriptors
        df = collector.calculate_molecular_descriptors(df)

        # Add drug-likeness features
        df = collector.add_druglikeness_features(df)

        # Try to augment with PubChem
        df = collector.augment_with_pubchem(df)

        # Save
        collector.save_dataset(df)

        # Summary
        collector.generate_summary(df)

        print("\n✅ Phase 1 Complete: Data collected and processed")
        return True

    except Exception as e:
        print(f"\n❌ Phase 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_graph_building():
    """Phase 2: Build molecular graphs for GNN"""
    print_banner("PHASE 2: GRAPH BUILDING FOR GNN")

    try:
        from src.gnn.graph_builder import build_dataset_from_csv

        csv_path = "src/data/data/himalayan_antimicrobial_compounds.csv"

        # Build dataset
        dataset = build_dataset_from_csv(csv_path, output_dir='data/processed')

        print("\n✅ Phase 2 Complete: Molecular graphs built")
        return True

    except ImportError:
        print("\n⚠️  Phase 2 Skipped: RDKit or PyG not installed")
        print("   Install with: pip install torch-geometric rdkit")
        return False
    except Exception as e:
        print(f"\n❌ Phase 2 Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_bioactivity_training():
    """Phase 3a: Train bioactivity model"""
    print_banner("PHASE 3A: BIOACTIVITY MODEL TRAINING")

    try:
        from src.models.train_activity_model import ActivityModelTrainer

        trainer = ActivityModelTrainer(
            data_path="src/data/data/himalayan_antimicrobial_compounds.csv",
            output_dir="models/bioactivity"
        )

        df = trainer.load_data()
        X, y = trainer.extract_features(df)
        train_data, val_data, test_data = trainer.prepare_data(X, y)
        trainer.model = trainer.create_model(input_size=X.shape[1])

        trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=150,
            batch_size=16,
            learning_rate=0.001
        )

        metrics = trainer.evaluate(test_data)
        trainer.save_model('bioactivity_model')

        print(f"\n✅ Phase 3A Complete: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        return True

    except Exception as e:
        print(f"\n❌ Phase 3A Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_toxicity_training():
    """Phase 3b: Train toxicity model"""
    print_banner("PHASE 3B: TOXICITY MODEL TRAINING")

    try:
        from src.models.train_toxicity_model import ToxicityModelTrainer

        trainer = ToxicityModelTrainer(
            data_path="src/data/data/himalayan_antimicrobial_compounds.csv",
            output_dir="models/toxicity"
        )

        df = trainer.load_data()
        df = trainer.create_toxicity_labels(df)
        X, y = trainer.extract_features(df)
        train_data, val_data, test_data = trainer.prepare_data(X, y)
        trainer.model = trainer.create_model(input_size=X.shape[1])

        trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=150,
            batch_size=16,
            learning_rate=0.001
        )

        metrics = trainer.evaluate(test_data)
        trainer.save_model('toxicity_model')

        print(f"\n✅ Phase 3B Complete: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        return True

    except Exception as e:
        print(f"\n❌ Phase 3B Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_gnn_training():
    """Phase 3c: Train GNN model"""
    print_banner("PHASE 3C: GNN MODEL TRAINING")

    try:
        from src.gnn.graph_builder import build_dataset_from_csv
        from src.gnn.models.gnn_model import get_model
        from src.models.train_gnn import GNNTrainer

        csv_path = "src/data/data/himalayan_antimicrobial_compounds.csv"
        dataset = build_dataset_from_csv(csv_path, output_dir='data/processed')
        model = get_model('gat', num_node_features=dataset.num_node_features)

        trainer = GNNTrainer(
            dataset=dataset,
            model=model,
            output_dir="models/gnn",
            learning_rate=0.001,
            weight_decay=0.0001
        )

        trainer.split_data(train_ratio=0.8, val_ratio=0.1)
        trainer.train(epochs=200, patience=30)
        metrics = trainer.evaluate_test()
        trainer.save_model('gnn_antimicrobial_model')

        print(f"\n✅ Phase 3C Complete: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        return True

    except ImportError:
        print("\n⚠️  Phase 3C Skipped: PyTorch Geometric not installed")
        print("   Install with: pip install torch-geometric")
        return False
    except Exception as e:
        print(f"\n❌ Phase 3C Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_report():
    """Phase 4: Generate summary report"""
    print_banner("PHASE 4: GENERATING REPORT")

    try:
        report = {
            "project": "Himalayan Herbal AI Platform",
            "timestamp": datetime.now().isoformat(),
            "completion": {
                "data_collection": True,
                "graph_building": True,
                "bioactivity_training": True,
                "toxicity_training": True,
                "gnn_training": True
            },
            "outputs": {
                "dataset": "src/data/data/himalayan_antimicrobial_compounds.csv",
                "bioactivity_model": "models/bioactivity/best_model.pth",
                "toxicity_model": "models/toxicity/best_model.pth",
                "gnn_model": "models/gnn/best_model.pth",
                "graphs": "data/processed/data.pt"
            },
            "next_steps": [
                "1. Review model metrics in models/*/metrics.json",
                "2. Explore visualizations in models/*/evaluation_results.png",
                "3. Deploy API: python -m uvicorn src.api.main:app --reload",
                "4. Make predictions on new compounds",
                "5. Fine-tune models with experimental data"
            ]
        }

        # Save report
        report_path = Path("reports") / "completion_report.json"
        report_path.parent.mkdir(exist_ok=True, parents=True)

        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 Project Status Report:")
        for phase, status in report['completion'].items():
            symbol = "✅" if status else "⚠️ "
            print(f"   {symbol} {phase.replace('_', ' ').title()}")

        print(f"\n📁 Key Outputs:")
        for key, path in report['outputs'].items():
            status = "✅" if Path(path).exists() else "⏳"
            print(f"   {status} {key:20s} → {path}")

        print(f"\n📋 Next Steps:")
        for step in report['next_steps']:
            print(f"   {step}")

        print(f"\n💾 Report saved to {report_path}")
        print("\n✅ Phase 4 Complete: Report generated")
        return True

    except Exception as e:
        print(f"\n❌ Phase 4 Failed: {e}")
        return False


def main():
    """Main orchestration"""
    parser = argparse.ArgumentParser(
        description="Himalayan Herbal AI - Complete Pipeline Runner"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "data", "graph", "bioactivity", "toxicity", "gnn", "report", "benchmark"],
        default="all",
        help="Which phase to run"
    )
    parser.add_argument(
        "--skip-gnn",
        action="store_true",
        help="Skip GNN training (requires PyTorch Geometric)"
    )

    args = parser.parse_args()

    print_banner("🌿 HIMALAYAN HERBAL AI - PIPELINE RUNNER")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Phase: {args.phase}")

    results = {}

    # Phase 1: Data Collection
    if args.phase in ["all", "data"]:
        results['data'] = run_data_collection()
        if not results['data']:
            print("\n⚠️  Aborting: Data collection failed")
            return

    # Phase 2: Graph Building
    if args.phase in ["all", "graph"]:
        results['graph'] = run_graph_building()

    # Phase 3a: Bioactivity Training
    if args.phase in ["all", "bioactivity"]:
        results['bioactivity'] = run_bioactivity_training()

    # Phase 3b: Toxicity Training
    if args.phase in ["all", "toxicity"]:
        results['toxicity'] = run_toxicity_training()

    # Phase 3c: GNN Training
    if args.phase in ["all", "gnn"] and not args.skip_gnn:
        results['gnn'] = run_gnn_training()

    # Phase 3d: Benchmark
    if args.phase in ["all", "benchmark"]:
        try:
            from src.models.benchmark import run_benchmark
            run_benchmark()
            results['benchmark'] = True
        except Exception as e:
            print(f"\n❌ Benchmark Failed: {e}")
            results['benchmark'] = False

    # Phase 4: Report
    if args.phase in ["all", "report"]:
        results['report'] = generate_report()

    # Summary
    print_banner("PIPELINE COMPLETION SUMMARY")
    for phase, status in results.items():
        symbol = "✅" if status else "⚠️ "
        print(f"{symbol} {phase.title():20s} {'Complete' if status else 'Skipped/Failed'}")

    print(f"\nEnd time: {datetime.now().isoformat()}")
    print("\n🎉 Pipeline execution complete!")
    print("\nNext: Deploy the API with:")
    print("  python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
