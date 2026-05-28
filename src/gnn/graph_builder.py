"""
Molecular Graph Builder for GNN
=================================

Converts SMILES strings to PyTorch Geometric graph objects.
Each molecule becomes a graph where:
- Nodes = atoms (with features like atom type, charge, etc.)
- Edges = chemical bonds (with features like bond type)
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Dict, Tuple, Optional
import os
import pickle


class MolecularGraphBuilder:
    """Converts molecular SMILES to PyG graphs"""
    
    def __init__(self):
        # Atom feature dimensions
        self.atom_features = {
            'symbol': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'Other'],
            'degree': [0, 1, 2, 3, 4, 5],  # Number of bonds
            'formal_charge': [-2, -1, 0, 1, 2],
            'num_hs': [0, 1, 2, 3, 4],  # Implicit hydrogens
            'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Other'],
            'is_aromatic': [False, True],
            'is_in_ring': [False, True]
        }
        
        # Bond feature dimensions
        self.bond_features = {
            'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
            'conjugated': [False, True],
            'in_ring': [False, True]
        }
        
    def smiles_to_graph(self, smiles: str, label: int) -> Optional[Data]:
        """
        Convert a SMILES string to a PyTorch Geometric Data object
        
        Args:
            smiles: SMILES string representation of molecule
            label: Binary label (1=antimicrobial active, 0=inactive)
            
        Returns:
            PyG Data object or None if conversion fails
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for better representation
            mol = Chem.AddHs(mol)
            
            # Extract atom features
            node_features = []
            for atom in mol.GetAtoms():
                node_features.append(self._get_atom_features(atom))
            
            # Extract bond features and connectivity
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                # Add edge in both directions (undirected graph)
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                
                bond_feat = self._get_bond_features(bond)
                edge_features.append(bond_feat)
                edge_features.append(bond_feat)  # Same features for both directions
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # Handle molecules with no bonds
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, len(self.bond_features['bond_type']) * 2 + 2), dtype=torch.float)
            
            y = torch.tensor([label], dtype=torch.long)
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                smiles=smiles
            )
            
            return data
            
        except Exception as e:
            print(f"Error processing SMILES {smiles[:50]}...: {str(e)}")
            return None
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract features from an atom"""
        from rdkit import Chem
        
        features = []
        
        # Atom type (one-hot encoding)
        symbol = atom.GetSymbol()
        features += self._one_hot(symbol, self.atom_features['symbol'])
        
        # Degree (number of bonds)
        degree = atom.GetDegree()
        features += self._one_hot(degree, self.atom_features['degree'])
        
        # Formal charge
        charge = atom.GetFormalCharge()
        features += self._one_hot(charge, self.atom_features['formal_charge'])
        
        # Number of implicit hydrogens
        num_hs = atom.GetTotalNumHs()
        features += self._one_hot(num_hs, self.atom_features['num_hs'])
        
        # Hybridization
        hybrid = str(atom.GetHybridization())
        features += self._one_hot(hybrid, self.atom_features['hybridization'])
        
        # Aromaticity
        is_aromatic = atom.GetIsAromatic()
        features += self._one_hot(is_aromatic, self.atom_features['is_aromatic'])
        
        # In ring
        is_in_ring = atom.IsInRing()
        features += self._one_hot(is_in_ring, self.atom_features['is_in_ring'])
        
        # Additional continuous features
        features.append(atom.GetMass() / 100.0)  # Normalized atomic mass
        
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract features from a bond"""
        from rdkit import Chem
        
        features = []
        
        # Bond type
        bond_type = str(bond.GetBondType())
        features += self._one_hot(bond_type, self.bond_features['bond_type'])
        
        # Conjugated
        is_conjugated = bond.GetIsConjugated()
        features += self._one_hot(is_conjugated, self.bond_features['conjugated'])
        
        # In ring
        is_in_ring = bond.IsInRing()
        features += self._one_hot(is_in_ring, self.bond_features['in_ring'])
        
        return features
    
    def _one_hot(self, value, options: List) -> List[float]:
        """One-hot encode a value"""
        encoding = [0.0] * len(options)
        
        if value in options:
            index = options.index(value)
            encoding[index] = 1.0
        else:
            # Use last position for unknown values
            encoding[-1] = 1.0
        
        return encoding
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """Calculate the total dimensionality of node and edge features"""
        
        node_dim = (
            len(self.atom_features['symbol']) +
            len(self.atom_features['degree']) +
            len(self.atom_features['formal_charge']) +
            len(self.atom_features['num_hs']) +
            len(self.atom_features['hybridization']) +
            len(self.atom_features['is_aromatic']) +
            len(self.atom_features['is_in_ring']) +
            1  # atomic mass
        )
        
        edge_dim = (
            len(self.bond_features['bond_type']) +
            len(self.bond_features['conjugated']) +
            len(self.bond_features['in_ring'])
        )
        
        return node_dim, edge_dim


class AntimicrobialDataset(Dataset):
    """PyTorch Geometric Dataset for antimicrobial compounds"""
    
    def __init__(self, csv_file: str, root: str = 'data/processed', 
                 transform=None, pre_transform=None):
        """
        Args:
            csv_file: Path to CSV with compounds and labels
            root: Directory to save processed data
            transform: Optional transform to apply on-the-fly
            pre_transform: Optional transform to apply during processing
        """
        self.csv_file = csv_file
        self.graph_builder = MolecularGraphBuilder()
        self.data_list = []
        
        super().__init__(root, transform, pre_transform)
    
    def __len__(self) -> int:
        """Return the number of graphs in the dataset"""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        """Get a single graph by index"""
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files"""
        return [os.path.basename(self.csv_file)]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files"""
        return ['data_list.pkl']
    
    def download(self):
        """Download is not needed - we already have the CSV"""
        pass
    
    def process(self):
        """Process the CSV into graph objects"""
        print("\n🔄 Processing molecules into graphs...")
        
        # Read the CSV
        df = pd.read_csv(self.csv_file)
        
        data_list = []
        failed = 0
        
        for idx, row in df.iterrows():
            smiles = row['smiles']
            
            # Skip if SMILES is NaN or not a string
            if pd.isna(smiles) or not isinstance(smiles, str):
                print(f"   ⚠️ Skipping row {idx}: invalid SMILES (type: {type(smiles)})")
                failed += 1
                continue
            
            smiles = str(smiles).strip()  # Ensure string and remove whitespace
            
            # Skip empty SMILES
            if not smiles:
                print(f"   ⚠️ Skipping row {idx}: empty SMILES")
                failed += 1
                continue
                
            label = int(row['antimicrobial_active'])
            
            # Convert to graph
            graph = self.graph_builder.smiles_to_graph(smiles, label)
            
            if graph is not None:
                # Add molecule name for tracking
                graph.compound_name = row.get('compound_name', f'compound_{idx}')
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                data_list.append(graph)
            else:
                failed += 1
            
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(df)} molecules...")
        
        print(f"\n✅ Successfully converted {len(data_list)} molecules to graphs")
        if failed > 0:
            print(f"⚠️  Failed to convert {failed} molecules")
        
        # Save processed data
        import pickle
        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(data_list, f)
        
        self.data_list = data_list
        print(f"💾 Saved {len(data_list)} graphs to {self.processed_paths[0]}")
    
    def load(self):
        """Load pre-processed data"""
        import pickle
        with open(self.processed_paths[0], 'rb') as f:
            self.data_list = pickle.load(f)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.data_list:
            self.load()
        
        stats = {
            'num_graphs': len(self.data_list),
            'num_features': self.data_list[0].num_node_features if self.data_list else 0,
            'num_edge_features': self.data_list[0].num_edge_features if self.data_list else 0,
            'num_classes': 2,
            'avg_nodes': np.mean([data.num_nodes for data in self.data_list]) if self.data_list else 0,
            'avg_edges': np.mean([data.num_edges for data in self.data_list]) if self.data_list else 0,
            'class_distribution': {}
        }
        
        # Count labels
        labels = [data.y.item() for data in self.data_list]
        stats['class_distribution']['inactive'] = labels.count(0)
        stats['class_distribution']['active'] = labels.count(1)
        
        return stats


def build_dataset_from_csv(csv_path: str, output_dir: str = 'data/processed') -> AntimicrobialDataset:
    """
    Main function to build PyG dataset from CSV
    
    Args:
        csv_path: Path to the CSV file with compounds
        output_dir: Directory to save processed graphs
        
    Returns:
        AntimicrobialDataset ready for GNN training
    """
    print("="*70)
    print("🔬 Building Graph Dataset for GNN")
    print("="*70)
    
    # Create dataset
    dataset = AntimicrobialDataset(csv_path, root=output_dir)
    
    # Print statistics
    stats = dataset.get_statistics()
    
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    print(f"Number of graphs: {stats['num_graphs']}")
    print(f"Node features: {stats['num_features']}")
    print(f"Edge features: {stats['num_edge_features']}")
    print(f"Average nodes per graph: {stats['avg_nodes']:.1f}")
    print(f"Average edges per graph: {stats['avg_edges']:.1f}")
    print(f"\nClass distribution:")
    print(f"  Inactive (0): {stats['class_distribution']['inactive']}")
    print(f"  Active (1): {stats['class_distribution']['active']}")
    print(f"  Imbalance ratio: {stats['class_distribution']['active']/stats['class_distribution']['inactive']:.2f}")
    
    print("\n✅ Dataset ready for GNN training!")
    
    return dataset


def main():
    """Example usage"""
    
    # Path to your CSV
    csv_path = '../data/data/himalayan_antimicrobial_compounds.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please run the data collector first!")
        return
    
    # Build the dataset
    dataset = build_dataset_from_csv(csv_path)
    
    # Show example graph
    print("\n" + "="*70)
    print("🔍 EXAMPLE GRAPH")
    print("="*70)
    example = dataset[0]
    print(f"Compound: {example.compound_name}")
    print(f"SMILES: {example.smiles}")
    print(f"Number of atoms (nodes): {example.num_nodes}")
    print(f"Number of bonds (edges): {example.num_edges // 2}")  # Divide by 2 because undirected
    print(f"Node feature dimension: {example.x.shape}")
    print(f"Edge feature dimension: {example.edge_attr.shape}")
    print(f"Label (antimicrobial): {example.y.item()}")
    
    print("\n" + "="*70)
    print("📋 Next Steps:")
    print("   1. The graphs are saved in data/processed/")
    print("   2. Ready to build GNN model")
    print("   3. Move to Phase 3: Model Training")
    print("="*70)


if __name__ == "__main__":
    main()