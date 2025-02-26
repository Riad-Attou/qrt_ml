import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from torch.utils.data import DataLoader, TensorDataset


def create_network(input_dim, hidden_dim=64):
    """Create a neural network for survival prediction using functional approach"""
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 1),
    )
    return model


def negative_log_likelihood(risk_pred, survival_time, event):
    """
    Custom loss function for survival analysis (Cox Partial Likelihood)

    Args:
        risk_pred: Model predictions (higher means higher risk)
        survival_time: Time to event or censoring
        event: Event indicator (1=event occurred, 0=censored)
    """
    # Sort in descending order of survival time
    idx = torch.argsort(survival_time, descending=True)
    risk_pred = risk_pred[idx]
    event = event[idx]

    # Compute log of sum of exp(risk) for all patients at risk at each time point
    hazard = torch.exp(risk_pred)
    log_risk = torch.log(torch.cumsum(hazard, dim=0) + 1e-8)

    # Select events only
    event_idx = (event == 1).nonzero().squeeze()
    if len(event_idx.shape) == 0:  # If only one event
        event_idx = event_idx.unsqueeze(0)

    # Compute the negative log likelihood
    neg_likelihood = -torch.mean(risk_pred[event_idx] - log_risk[event_idx])
    return neg_likelihood


def modele_survival_nn(cas, merged_train, merged_test, target_df):
    """
    Neural network survival model for predicting risk scores using a functional approach

    Args:
        cas: Case option (1 for train/test split, 2 for using all training data)
        merged_train: Training data with features
        merged_test: Test data with features
        target_df: Target data with survival times and events

    Returns:
        train_ci_ipcw: Concordance index on training data
    """
    features = [
        "BM_BLAST",
        "HB",
        "PLT",
        "WBC",
        "ANC",
        "nb_del",
        "nb_t",
        "nb_dup",
        "nb_inv",
        "complex_karyotype",
        "nb_philadelphia",
        "cyto_risk_defavorable",
        "cyto_risk_intermediaire",
        "CLASSE_CYT",
        "gene_risk_favorable",
        "gene_risk_defavorable",
        "max_VAF",
        "stop_gained_count",
        "frameshift_variant_count",
        "non_synonymous_count",
        "weighted_stop_gained",
        "weighted_frameshift",
        "weighted_non_synonymous",
        "min_depth",
        "flt3_mutated",
        "npm1_mutated",
        "pire_mutations",
        "dnmt3a_mutated",
        "asxl1_mutated",
        "total_mutations",
        "tca_ref",
        "gata2_mutated",
        "ezh2_mutated",
        "crebbp_mutated",
        "zrsr2_mutated",
        "nbr_mono",
        "bcor_mutated",
        "runx1_mutated",
        "stag2_mutated",
        "abl1_mutated",
        "nfe2_mutated",
        "ddx41_mutated",
        "csnk1a1_mutated",
        "sh2b3_mutated",
        "gene_risk_pondere",
        "tp53_mutated",
        "srsf2_mutated",
        "u2af1_mutated",
        "nras_mutated",
        "gnb1_mutated",
        "csf3r_mutated",
        "mpl_mutated",
        "hax1_mutated",
        "rit1_mutated",
        "smc3_mutated",
        "wt1_mutated",
        "atm_mutated",
        "cbl_mutated",
        "etv6_mutated",
        "etnk1_mutated",
        "kras_mutated",
        "arid2_mutated",
        "ptpn11_mutated",
        "brca2_mutated",
        "pds5b_mutated",
        "idh2_mutated",
        "nf1_mutated",
        "ppm1d_mutated",
        "cebpa_mutated",
        "idh1_mutated",
        "myd88_mutated",
        "kit_mutated",
        "phf6_mutated",
        "bcorl1_mutated",
        "jak2_mutated",
        "cux1_mutated",
        "vegfa_mutated",
        "mll_mutated",
    ]

    # Prepare target data
    target_df = target_df.dropna(subset=["OS_YEARS", "OS_STATUS"])

    # Prepare feature data
    X = merged_train.loc[merged_train["ID"].isin(target_df["ID"]), features]

    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare target variables
    y_df = target_df[["OS_STATUS", "OS_YEARS"]]
    y_df = y_df.loc[y_df.index.isin(X.index)]

    times = y_df["OS_YEARS"].values
    events = y_df["OS_STATUS"].values.astype(int)  # Convert bool to int

    if cas == 1:
        # Split data into training and testing sets
        X_train, X_test, times_train, times_test, events_train, events_test = (
            train_test_split(
                X_scaled, times, events, test_size=0.2, random_state=42
            )
        )
        y_train = Surv.from_arrays(events_train, times_train)
        y_test = Surv.from_arrays(events_test, times_test)
    else:
        X_train = X_scaled
        times_train = times
        events_train = events
        y_train = Surv.from_arrays(events, times)

        # Scale test data
        X_test = scaler.transform(merged_test[features])
        y_test = None

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    times_train_tensor = torch.FloatTensor(times_train)
    events_train_tensor = torch.FloatTensor(events_train)

    # Create DataLoader for batch training
    train_dataset = TensorDataset(
        X_train_tensor, times_train_tensor, events_train_tensor
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create the network
    input_dim = X_train.shape[1]
    model = create_network(input_dim)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for X_batch, times_batch, events_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            risk_pred = model(X_batch).squeeze()

            # Compute loss
            loss = negative_log_likelihood(
                risk_pred, times_batch, events_batch
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.6f}"
            )

    # Get predictions
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_tensor).squeeze().numpy()
        X_test_tensor = torch.FloatTensor(X_test)
        pred_test = model(X_test_tensor).squeeze().numpy()

    # For survival analysis, higher risk score = worse prognosis
    # So we need to negate the predictions to get the correct risk scores
    risk_score = pred_test

    # Create results dataframe
    df_risk_score = pd.DataFrame(risk_score, columns=["risk_score"])
    df_risk_score.insert(0, "ID", merged_test["ID"])
    df_risk_score.to_csv("pred_test_nn.csv", index=False)

    # Evaluate model with concordance index
    if cas == 1:
        train_ci_ipcw = concordance_index_ipcw(
            y_train, y_train, -pred_train, tau=7
        )[0]
        test_ci_ipcw = concordance_index_ipcw(
            y_test, y_test, -pred_test, tau=7
        )[0]
        print(
            f"Neural Network Survival Model Concordance Index IPCW on train: {train_ci_ipcw:.6f}"
        )
        print(
            f"Neural Network Survival Model Concordance Index IPCW on test: {test_ci_ipcw:.6f}"
        )
    else:
        train_ci_ipcw = concordance_index_ipcw(
            y_train, y_train, -pred_train, tau=7
        )[0]
        print(
            f"Neural Network Survival Model Concordance Index IPCW on train: {train_ci_ipcw:.6f}"
        )

    return train_ci_ipcw


# Function to integrate with the existing pipeline
def main_neural_network():
    # Import the existing functions from the main code
    from main_script import charger_donnees, traitement_donnees

    # Set the number of classes for feature engineering
    nbclasses = 100

    # Load data using the existing function
    df_train, df_eval, mol_df, mol_eval, target_df = charger_donnees()

    # Prepare data using the existing function
    merged_train, merged_test = traitement_donnees(
        nbclasses, df_train, df_eval, mol_df, mol_eval, target_df
    )

    # Run the neural network model
    modele_survival_nn(2, merged_train, merged_test, target_df)


if __name__ == "__main__":
    main_neural_network()
