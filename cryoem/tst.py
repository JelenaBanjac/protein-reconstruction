
def train_angle_recovery(steps, batch_size, projection_idx, 
                        angles_predicted, 
                        est_dist_input, est_dist, 
                        learning_rate=0.01, 
                        optimization=False):
    if est_dist_input.shape[1] == 3:
        convert = euler2quaternion
        dt_type = "dQ"
    else:
        convert = lambda x: x 
        dt_type = "dP"
    
    losses = np.empty(steps)
    time_start = time()
    optimizer = Adam(learning_rate=learning_rate)
    
    for step, idx1, idx2 in sample_iter(projection_idx, batch_size, style="random"):

        a1 = [angles_predicted[i] for i in idx1]
        a2 = [angles_predicted[i] for i in idx2]

        # Compute distances
        in1 = convert([est_dist_input[i] for i in idx1])
        in2 = convert([est_dist_input[i] for i in idx2])
        
        distance_target = est_dist(in1, in2)

        # Optimize by gradient descent.
        if optimization:
            losses[step-1], gradients = gradient(a1, a2, distance_target, dt_type=dt_type, space="dQspace")
            optimizer.apply_gradients(zip(gradients, a1 + a2))
        else:
            losses[step-1] = loss(a1, a2, distance_target, dt_type=dt_type, space="dQspace")
        
        # Periodically report progress.
        if ((step % (steps//10)) == 0) or (step == steps):
            time_elapsed = time() - time_start
            print(f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}')

    if optimization:
        # Plot convergence.
        sns.set(style="white", color_codes=True)
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(15,10))
        sns.lineplot(x=np.linspace(0, time()-time_start, steps), y=losses, ax=ax, marker="o")
        ax.set(xlabel='time [s]', ylabel='loss')
        plt.show()
    else:
        print(f"Mean loss: {np.mean(losses)}")
