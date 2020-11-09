import SimpleITK as sitk 

def gaussian_filtering(img_in, sigma=2.,  rescale=True):
    """apply Gaussia filter to input image

    Args:
        img_in (sitk image): input image
        sigma (float) : width of the filter
        rescale (bool, optional): rescale the output image. Defaults to True.

    Returns:
        sitk image: output image
    """
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(sigma)
    img_out = gaussian.Execute(img_in)

    return _rescale(img_out, rescale)


def otsu_filtering(img_in, rescale=True):
    """apply Otsu filter to input image

    Args:
        img_in (sitk image): input image
        rescale (bool, optional): rescale the output image. Defaults to True.

    Returns:
        sitk image: output image
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    img_out = otsu_filter.Execute(img_in)

    return _rescale(img_out, rescale)

def binary_threshold(img_in, lowerThreshold=100, upperThreshold=150, insideValue=1, outsideValue=0):
    """binary threshold

    Args:
        img_in ([type]): [description]
        lowerThreshold (int, optional): [description]. Defaults to 100.
        upperThreshold (int, optional): [description]. Defaults to 150.
        insideValue (int, optional): [description]. Defaults to 1.
        outsideValue (int, optional): [description]. Defaults to 0.
        rescale (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    img_out = sitk.BinaryThreshold(img_in, 
                                   lowerThreshold=lowerThreshold, 
                                   upperThreshold=upperThreshold, 
                                   insideValue=insideValue, 
                                   outsideValue=outsideValue)
    return img_out

def otsu_threshold(img_in, insideValue=1, outsideValue=0):
    """otsu thresholding

    Args:
        ing_in ([type]): [description]
        rescale (bool, optional): [description]. Defaults to True.
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(insideValue)
    otsu_filter.SetOutsideValue(outsideValue)
    img_out = otsu_filter.Execute(img_in)
    
    return img_out

def confidence_connected(img_in, seed, numberOfIterations=1, multiplier=2.5, initialNeighborhoodRadius=1, replaceValue=1 ):
    return sitk.ConfidenceConnected(img_in, seedList=[seed],
                                   numberOfIterations=numberOfIterations,
                                   multiplier=multiplier,
                                   initialNeighborhoodRadius=initialNeighborhoodRadius,
                                   replaceValue=replaceValue)
    

def _rescale(img, rescale):
    """rescale the image if necessary."""
    if rescale:
        img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
    return img
