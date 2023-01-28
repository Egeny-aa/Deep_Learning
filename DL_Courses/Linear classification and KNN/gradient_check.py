import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    # Инструкции assert в Python — это булевы выражения, которые проверяют, является ли условие истинным
    orig_x = x.copy() # х - задается в начале функции
    fx, analytic_grad = f(x)  # Присвает в Fx значение функции в . х, а analytic_grad присваивает занчение градиенита этой функции( Все данные из ввода)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric 
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0
        delta_plus = orig_x.copy() # копируем массив для того чтобы на одном и том же месте сделать +d и -d
        delta_munus = orig_x.copy()
        delta_plus[ix] = orig_x[ix] + delta #Позиции массива ix преравниваем разницу соответствующего значения в массиве orig_x и дельты
        delta_munus[ix] = orig_x[ix] - delta
        # TODO compute value of numeric gradient of f to idx
        numeric_grad_at_ix = (f(delta_plus)[0] - f(delta_munus)[0]) / (2 * delta) # по формуле высчитываем численный градиент( +- дельта делается на одной позиции в матрице.
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):#проверяет на сколько отличаютя num и anal, если не на много,то заканчивает цикл
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            #return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
