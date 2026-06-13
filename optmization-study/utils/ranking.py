def rank_optimizers(results):

    ranking = []

    for opt, data in results.items():
        final_acc = data["mean_acc"][-1]
        convergence = data["convergence_epoch"]

        ranking.append((opt, final_acc, convergence))

    # sort by best accuracy, then fastest convergence
    ranking.sort(key=lambda x: (-x[1], x[2]))

    return ranking