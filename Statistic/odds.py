# value(profit)
def minimum_odd(score):
    '''
    value = score.mean*Aporte*(odd-1) - (1-score.mean)*Aporte > 0
    '''
    odd_min = (1-score.mean())/(score.mean()) + 1
    return odd_min

# kelly criterion
def kelly_criterion(score, odd):
    kelly_criterion = (score.mean()*odd - 1)/(odd -1)
    return kelly_criterion