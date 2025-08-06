import math
import logging

from generator import generate_students
from loss import gnc_sensing_loss, gd_sensing_loss
from model import DiagonalSSM

import torch
from torch.optim import Adam
from torch.optim import SGD

def train_gd(
        seed: int,
        student_dim: int,
        device: torch.device,
        y_teacher: torch.Tensor,
        dataset: torch.Tensor,
        init_scale: float,
        lr: float,
        epochs: int,
        calc_loss_only_on_last_output: bool = True,
        optimizer: str = "adam",
    ):
    """
    Trains only the diagonal transition matrix A of the student SSM.
    B and C stay fixed to 1s as defined inside DiagonalSSM.
    """

    # --- build student -------------------------------------------------------
    _, _, input_dim = dataset.shape
    output_dim = y_teacher.shape[-1]
    model = DiagonalSSM(state_dim=student_dim,
                        input_dim=input_dim,
                        output_dim=output_dim,  # C‘s output size
                        init_scale=init_scale
                        ).to(device)

    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer == "gd":
        optimizer = SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")

    # log index of A_j that is the largest
    max_A_j_idx = torch.argmax(model.A_diag)
    logging.info(f"initial model: max A_j index: {max_A_j_idx}")
    logging.info(f"initial model: A values are {model.A_diag.cpu().tolist()}")

    # --- keep track of losses ------------------------------------------------
    train_hist, test_hist = [], []
    dataset = dataset.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # forward pass on the full sequence
        y_pred = model(dataset)
        loss_train, loss_gen = gd_sensing_loss(y_pred=y_pred,
                                               y_teacher=y_teacher,
                                               calc_loss_only_on_last_output=calc_loss_only_on_last_output)
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            # save losses
            train_hist.append(loss_train.item())
            if epoch % 10 == 0:  # test every 10 epochs
                test_hist.append(loss_gen.item())

    max_A_j_idx = torch.argmax(model.A_diag)
    logging.info(f"final model: max A_j index: {max_A_j_idx}")
    logging.info(f"final model: A values are {model.A_diag.cpu().tolist()}")
    logging.info(f"final model: average A value: {model.A_diag.mean().item()}")
    logging.info(f"final model: variance of A values: {model.A_diag.var().item()}")
    logging.info(f"train loss is {train_hist[-1]}")
    logging.info(f"impulse response loss is {test_hist[-1]}")
    return test_hist[-1] if test_hist else float("nan"), train_hist[-1] if train_hist else float("nan")


def train_gnc(seed: int,
             student_dim: int,
             device: torch.device,
             y_teacher: torch.Tensor,
             dataset: torch.Tensor,
             eps_train: float,
             num_samples: int,
             batch_size: int,
             sequence_length: int,
             calc_loss_only_on_last_output=True,
              ):
    prior_gen_losses = []
    gnc_gen_losses = []
    eps_train_by_dim = eps_train / student_dim

    for batch in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - batch * batch_size)
        students = generate_students(student_dim, bs, sequence_length, device)
        train_losses, gen_losses = gnc_sensing_loss(students=students, y_teacher=y_teacher, x=dataset, calc_loss_only_on_last_output=calc_loss_only_on_last_output)
        succ_mask = train_losses < eps_train_by_dim
        prior_gen_losses.extend(gen_losses.cpu().tolist())
        succ_mask = succ_mask.squeeze(-1)
        gnc_gen_losses.extend(gen_losses[succ_mask].cpu().tolist())
    mean_prior = sum(prior_gen_losses) / len(prior_gen_losses)
    mean_gnc = sum(gnc_gen_losses) / len(gnc_gen_losses) if gnc_gen_losses else float("nan")
    if len(gnc_gen_losses) == 0:
        logging.info("No GNC sensing losses")
    return mean_prior, mean_gnc