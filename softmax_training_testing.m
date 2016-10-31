clear all
load firingrate.mat

alpha = 0.5;

% firingrate = firingrate_generated;
channel = size(firingrate, 1);
number_samples_per_direction = size(firingrate, 2);
number_directions = size(firingrate, 3);
train_samples_per_direction = round(0.9 * number_samples_per_direction);
test_samples_per_direction = number_samples_per_direction - train_samples_per_direction;
training_data = zeros(channel + 1, train_samples_per_direction * number_directions);
training_data(channel + 1, :) = ones(1, train_samples_per_direction * number_directions);
training_label = zeros(1, train_samples_per_direction * number_directions);
training_result_matrix = zeros(number_directions, train_samples_per_direction * number_directions);
training_data_count = 1;
testing_data = zeros(channel + 1, test_samples_per_direction * number_directions);
testing_data(channel + 1, :) = ones(1, test_samples_per_direction * number_directions);
testing_label = zeros(1, test_samples_per_direction * number_directions);
testing_result_matrix = zeros(number_directions, test_samples_per_direction * number_directions);
testing_data_count = 1;
for k = 1:number_directions
    for j = 1:number_samples_per_direction
        if j > train_samples_per_direction
            testing_data(1:channel, testing_data_count) = firingrate(:, j, k);
            testing_result_matrix(k, testing_data_count) = 1;
            testing_label(1, testing_data_count) = k;
            testing_data_count = testing_data_count + 1;
        else
            training_data(1:channel, training_data_count) = firingrate(:, j, k);
            training_result_matrix(k, training_data_count) = 1;
            training_label(1, training_data_count) = k;
            training_data_count = training_data_count + 1;
        end
    end
end
max_element = max(max(training_data));
training_data = training_data / max_element;
testing_data = testing_data / max_element;
A = rand(channel + 1, number_directions);
calculation_result = (training_data' * A)';
result = zeros(size(training_result_matrix));
for i = 1:size(calculation_result, 2)
    temp = calculation_result(:, i);
    temp = exp(temp);
    total = sum(temp);
    result(:, i) = temp / total;
end
[~, result_label] = max(result, [], 1);
initial_correct = size(find(result_label - training_label == 0), 2)
training_iterations = 1000;
error = [];
training_accuracy = [initial_correct / size(training_data, 2)];
calculation_result = (testing_data' * A)';
result = zeros(size(testing_result_matrix));
for i = 1:size(calculation_result, 2)
    temp = calculation_result(:, i);
    temp = exp(temp);
    total = sum(temp);
    result(:, i) = temp / total;
end
[~, result_label] = max(result, [], 1);
initial_correct = size(find(result_label - testing_label == 0), 2);
testing_accuracy = [initial_correct / size(testing_data, 2)];
for iter = 1:training_iterations
    iter
    calculation_result = training_data' * A;
    calculation_result = calculation_result';
    result = zeros(size(training_result_matrix));
    error_local = 0;
    for i = 1:size(calculation_result, 2)
        temp = calculation_result(:, i);
        temp = exp(temp);
        total = sum(temp);
        result(:, i) = temp / total;
        for j = 1:number_directions
            if (j == find(training_result_matrix(:, i) == 1))
                error_local = error_local - log(result(j, i));
            end
        end
    end
    error = [error error_local];
    delta_w = zeros(channel + 1, number_directions);
    for sample_number = 1:size(training_data, 2)
        for c = 1:(channel + 1)
            for direction = 1:number_directions
                gradient = result(direction, sample_number);
                if (direction == find(training_result_matrix(:, sample_number) == 1))
                    gradient = gradient - 1;
                end
                gradient = gradient * training_data(c, sample_number);
                delta_w(c, direction) = delta_w(c, direction) + gradient;
            end
        end
        
    end
    A(1:channel, :) = A(1:channel, :) - alpha * (delta_w(1:channel, :) / size(training_result_matrix, 2)); % + 0.0001 * A(1:95, :)
    A(channel + 1, :) = A(channel + 1, :) - alpha * delta_w(channel + 1, :) / size(training_result_matrix, 2);
    calculation_result = (training_data' * A)';
    result = zeros(size(training_result_matrix));
    for i = 1:size(calculation_result, 2)
        temp = calculation_result(:, i);
        temp = exp(temp);
        total = sum(temp);
        result(:, i) = temp / total;
        if (i == 735)
            result(:, i)
        end
    end
    [~, result_label] = max(result, [], 1);
    training_accuracy = [training_accuracy size(find(result_label - training_label == 0), 2) / size(training_data, 2)];
    figure(1), plot(error);
    hold on;
    
    test_calc = (testing_data' * A)'
    result = zeros(size(testing_result_matrix));
    for i = 1:size(test_calc, 2)
        temp = test_calc(:, i);
        temp = exp(temp);
        total = sum(temp);
        result(:, i) = temp / total;
    end
    [~, result_label] = max(result, [], 1);
    testing_accuracy = [testing_accuracy size(find(result_label - testing_label == 0), 2) / size(testing_data, 2)];
    figure(2), plot(training_accuracy, 'b');
    hold on;
    plot(testing_accuracy, 'r');
    drawnow
    
end
calculation_result = training_data' * A;
calculation_result = calculation_result';
error_local = 0;
for i = 1:size(calculation_result, 2)
    temp = calculation_result(:, i);
    temp = exp(temp);
    total = sum(temp);
    result(:, i) = temp / total;
    for j = 1:number_directions
        if (j == find(training_result_matrix(:, i) == 1))
            error_local = error_local - log(result(j, i));
        end
    end
end
error = [error error_local];
result = zeros(size(training_result_matrix));
for i = 1:size(calculation_result, 2)
    temp = calculation_result(:, i);
    temp = exp(temp);
    total = sum(temp);
    result(:, i) = temp / total;
end
[~, result_label] = max(result, [], 1);
size(find(result_label - training_label == 0), 2)
