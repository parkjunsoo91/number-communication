import matplotlib.pyplot as plt

def draw(epoch_history1, train_history, epoch_history2, validation_history):
    plt.figure(2)
    plt.clf()
    #durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch_history1, train_history, 'b')
    plt.plot(epoch_history2, validation_history, 'r')
    
    #plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    #if len(durations_t) >= 100:
    #    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #    means = torch.cat((torch.zeros(99), means))
    #    plt.plot(means.numpy())

    plt.pause(0.001)
