import torch

from ssm_forward import ssm_forward


def get_train_and_gen_from_y(y: torch.Tensor, calc_loss_only_on_last_output=True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the output tensor into training and generalization parts.

    Args:
        y: Output tensor (*batch_size, num_measurements, sequence_length, output_dim).
        calc_loss_only_on_last_output: Calculate loss only on last output

    Returns:
        Tuple containing:
            - y_train: Training output (batch_size, num_measurements, output_dim)
            - y_gen: Generalization output (batch_size, sequence_length, output_dim)
    """
    if y.dim() == 3:
        # If y is 3D, we assume it has shape (num_measurements, sequence_length, output_dim)
        y = y.unsqueeze(0) # Add batch dimension
    # y should now be (batch_size, num_measurements, sequence_length, output_dim)
    if calc_loss_only_on_last_output:
        if y.shape[1] == 1: # If there is only one measurement, we don't need to exclude the last measurement
            y_train = y[:, -1, -1, :]
        else:
            y_train = y[:, :-1, -1, :]  # Exclude the last measurement for training loss and get only the last output
    else:
        y_train = y[:, :-1, :, :]

    y_gen = y[:, -1, :, :]  # Generalization output
    return y_train, y_gen


def get_y_teacher(teacher: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    x: torch.Tensor) -> torch.Tensor:
        """
        Get the output of the teacher model for the given input x.

        Args:
            teacher: Tuple containing teacher matrices (A_teacher, B_teacher, C_teacher).
            x: Input sequence (num_measurements, sequence_length, input_dim).

        Returns:
            Output of the teacher model.
        """
        y_teacher = ssm_forward(*teacher, x)
        return y_teacher


def sensing_loss(y_teacher: torch.Tensor,
                 y_student: torch.Tensor,
                 calc_loss_only_on_last_output=True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the sensing loss for the student model.
        """
        y_teacher_train, y_teacher_gen = get_train_and_gen_from_y(y_teacher, calc_loss_only_on_last_output)
        y_student_train, y_student_gen = get_train_and_gen_from_y(y_student, calc_loss_only_on_last_output)
        # y_*_train: (batch_size, num_measurements, output_dim)
        # y_*_gen: (batch_size, sequence_length, output_dim)
        if calc_loss_only_on_last_output:
            if y_student_train.shape[1] == 1:
                mse_loss = torch.mean((y_student_train - y_teacher_train) ** 2, dim=(1))
            else:
                mse_loss = torch.mean((y_student_train - y_teacher_train) ** 2, dim=(1, 2)) # Mean squared error on training output
        else:
            mse_loss = torch.mean((y_student_train - y_teacher_train) ** 2, dim=(1, 2, 3))
        general_loss = torch.sum((y_student_gen - y_teacher_gen) ** 2, dim=(1, 2))  # Sum of squares on generalization output

        return mse_loss, general_loss


def gnc_sensing_loss(y_teacher: torch.Tensor,
                    students: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     x: torch.Tensor,
                     calc_loss_only_on_last_output: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    # Forward pass for all students in batch
    y_student = ssm_forward(*students, x) # (batch_size, num_measurements, sequence_length, 1)
    # Calculate sensing loss
    return sensing_loss(y_student=y_student, y_teacher=y_teacher, calc_loss_only_on_last_output=calc_loss_only_on_last_output)


def gd_sensing_loss(y_pred: torch.Tensor,
                    y_teacher: torch.Tensor,
                    calc_loss_only_on_last_output=True
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the sensing loss for gradient descent training.

    Args:
        y_pred: Predicted output from the model (batch_size, num_measurements, sequence_length, output_dim).
        y_teacher: Teacher's output (1, num_measurements, 1, output_dim).
        calc_loss_only_on_last_output: Calculate loss only on last output

    Returns:
        Tuple containing:
            - mse_loss: Mean squared error loss on training output.
            - general_loss: Sum of squares loss on generalization output.
    """
    return sensing_loss(y_student=y_pred, y_teacher=y_teacher, calc_loss_only_on_last_output=calc_loss_only_on_last_output)