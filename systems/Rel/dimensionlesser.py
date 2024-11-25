def theta(sigma):
    if sigma<1:
        return 0
    else:
        return 1

def get_alpha(sigma):
    return 1 / 2 - theta(sigma) / 6

def get_gamma(sigma):
    return theta(sigma)/3

def get_coeffs(sigma):
    alpha = get_alpha(sigma)
    gamma = get_gamma(sigma)
    s1 = sigma ** (gamma-alpha)
    s2 = sigma ** (gamma-1)
    s3 = sigma ** (2*alpha+gamma-1)
    return (s1,s2,s3)
    
def get_P(m,omega):
    alpha=get_alpha(omega/m)
    P=(m ** (alpha)) * (omega ** (1-alpha))
    return P 

def get_E(m,omega):
    gamma=get_gamma(omega/m)
    E=(m ** (gamma)) * (omega ** (1-gamma))
    return E
    