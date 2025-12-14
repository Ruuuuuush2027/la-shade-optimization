"""
Test script to verify planting opportunity constraint is working.

Tests:
1. PlantingOpportunityConstraint class directly
2. Integration with Approach1WeightedSum
3. Verify non-plantable locations get zero reward
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from base.constraints import PlantingOpportunityConstraint, ConstraintManager


def test_planting_constraint_class():
    """Test the PlantingOpportunityConstraint class directly."""
    print("\n" + "="*70)
    print("TEST 1: PlantingOpportunityConstraint Class")
    print("="*70)

    # Create test data
    test_data = pd.DataFrame({
        'planting_opportunity': [0.5, 1.5, 2.5, 3.5, 5.0],
        'latitude': [34.0] * 5,
        'longitude': [-118.0] * 5
    })

    # Test with default threshold (2.0)
    constraint = PlantingOpportunityConstraint()

    print(f"\nThreshold: {constraint.min_threshold}")
    print(f"Field: {constraint.field_name}")
    print(f"Hard constraint: {constraint.use_hard_constraint}")

    print("\n{:<20} {:<15} {:<10}".format("Planting Score", "Is Plantable?", "Penalty"))
    print("-" * 45)

    for idx, row in test_data.iterrows():
        is_plantable = constraint.is_plantable(row)
        penalty = constraint.get_planting_penalty(row)
        print(f"{row['planting_opportunity']:<20} {str(is_plantable):<15} {penalty:<10}")

    # Verify expected behavior
    assert constraint.get_planting_penalty(test_data.iloc[0]) == 0.0, "Score 0.5 should get 0 penalty"
    assert constraint.get_planting_penalty(test_data.iloc[1]) == 0.0, "Score 1.5 should get 0 penalty"
    assert constraint.get_planting_penalty(test_data.iloc[2]) == 1.0, "Score 2.5 should get full reward"
    assert constraint.get_planting_penalty(test_data.iloc[3]) == 1.0, "Score 3.5 should get full reward"

    print("\n✓ All assertions passed!")


def test_constraint_manager():
    """Test ConstraintManager with planting constraint."""
    print("\n" + "="*70)
    print("TEST 2: ConstraintManager Integration")
    print("="*70)

    # Create test data
    test_data = pd.DataFrame({
        'planting_opportunity': [1.0, 3.0, 5.0],
        'latitude': [34.0, 34.01, 34.02],
        'longitude': [-118.0, -118.01, -118.02],
        'lashade_tot1200': [0.1, 0.2, 0.3],
        'lashade_tot1500': [0.1, 0.2, 0.3],
        'lashade_tot1800': [0.1, 0.2, 0.3]
    })

    # Initialize manager with planting constraint
    manager = ConstraintManager(
        planting_config={'min_threshold': 2.0}
    )

    print("\nTesting penalties for 3 locations:")
    print("{:<10} {:<20} {:<20} {:<20}".format(
        "Index", "Planting Score", "Planting Penalty", "Shade Penalty"
    ))
    print("-" * 70)

    for idx in range(len(test_data)):
        features = test_data.iloc[idx]
        penalties = manager.get_all_penalties(
            state=[],
            action_idx=idx,
            min_distance=1.0,
            features=features
        )

        print(f"{idx:<10} {features['planting_opportunity']:<20} "
              f"{penalties['planting_penalty']:<20} "
              f"{penalties['existing_shade_penalty']:<20}")

    # Verify planting penalty is in the returned dict
    penalties = manager.get_all_penalties([], 0, 1.0, test_data.iloc[0])
    assert 'planting_penalty' in penalties, "planting_penalty should be in penalties dict"
    assert penalties['planting_penalty'] == 0.0, "Low planting score should give 0 penalty"

    print("\n✓ ConstraintManager test passed!")


def test_approach1_integration():
    """Test Approach1 with real data to verify planting constraint works end-to-end."""
    print("\n" + "="*70)
    print("TEST 3: EnhancedWeightedSumReward Integration")
    print("="*70)

    # Import here to avoid issues
    from approaches.approach1_weighted import EnhancedWeightedSumReward

    # Load real data
    data_path = '/home/fhliang/projects/libero_shade/shade_optimization_data_usc_simple_features.csv'

    try:
        df = pd.read_csv(data_path)
        print(f"\n✓ Loaded {len(df)} locations from dataset")

        # Check planting_opportunity stats
        if 'planting_opportunity' in df.columns:
            print(f"\nPlanting Opportunity Statistics:")
            print(f"  Min:  {df['planting_opportunity'].min():.2f}")
            print(f"  25%:  {df['planting_opportunity'].quantile(0.25):.2f}")
            print(f"  50%:  {df['planting_opportunity'].quantile(0.50):.2f}")
            print(f"  75%:  {df['planting_opportunity'].quantile(0.75):.2f}")
            print(f"  Max:  {df['planting_opportunity'].max():.2f}")

            # Find locations below threshold
            threshold = 2.0
            below_threshold = df[df['planting_opportunity'] <= threshold]
            print(f"\n  Locations below threshold ({threshold}): {len(below_threshold)}")
            print(f"  Locations above threshold: {len(df) - len(below_threshold)}")

        # Initialize Approach1 with planting constraint
        config = {
            'constraints': {
                'planting': {'min_threshold': 2.0, 'use_hard_constraint': True}
            }
        }

        approach = EnhancedWeightedSumReward(df, config=config, region='USC')
        print(f"\n✓ Initialized EnhancedWeightedSumReward with planting constraint")

        # Test on a few locations with different planting scores
        test_indices = [0, 100, 500, 1000]

        print(f"\nTesting rewards for locations with varying planting scores:")
        print("{:<10} {:<20} {:<15}".format("Index", "Planting Score", "Reward"))
        print("-" * 45)

        for idx in test_indices[:min(len(test_indices), len(df))]:
            planting_score = df.iloc[idx]['planting_opportunity']
            reward = approach.calculate_reward(state=[], action_idx=idx)
            print(f"{idx:<10} {planting_score:<20.2f} {reward:<15.4f}")

        # Verify that locations with low planting scores get 0 reward
        low_planting_idx = df[df['planting_opportunity'] <= 2.0].index[0]
        low_reward = approach.calculate_reward(state=[], action_idx=low_planting_idx)

        print(f"\n✓ Location {low_planting_idx} (planting={df.iloc[low_planting_idx]['planting_opportunity']:.2f}) "
              f"gets reward: {low_reward:.4f}")

        if low_reward == 0.0:
            print("✓ PASS: Non-plantable location correctly gets zero reward!")
        else:
            print(f"✗ FAIL: Expected 0.0 reward, got {low_reward}")

        # Test breakdown
        print("\nDetailed breakdown for low-planting location:")
        breakdown = approach.get_component_breakdown(state=[], action_idx=low_planting_idx)
        for key, value in breakdown.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    except FileNotFoundError:
        print(f"\n✗ Data file not found: {data_path}")
        print("  Skipping integration test")
    except Exception as e:
        print(f"\n✗ Error in integration test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING PLANTING OPPORTUNITY CONSTRAINT")
    print("="*70)

    test_planting_constraint_class()
    test_constraint_manager()
    test_approach1_integration()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
