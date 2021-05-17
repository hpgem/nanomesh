import SimpleITK as sitk


def gaussian_filter(img_in, sigma=2., rescale=True):
    """apply Gaussia filter to input image.

    Parameters
    ----------
    img_in : sitk image
        input image
    sigma : float
        width of the filter
    rescale : bool, optional
        rescale the output image. Defaults to True.

    Returns
    -------
    sitk image: output image
    """
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(sigma)
    img_out = gaussian.Execute(img_in)

    return _rescale(img_out, rescale)


def otsu_filter(img_in, rescale=True):
    """apply Otsu filter to input image.

    Parameters
    ----------
    img_in : sitk image
        input image
    rescale : bool, optional
        rescale the output image. Defaults to True.

    Returns
    -------
    sitk image: output image
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    img_out = otsu_filter.Execute(img_in)

    return _rescale(img_out, rescale)


def binary_threshold(img_in,
                     lowerThreshold=100,
                     upperThreshold=150,
                     insideValue=1,
                     outsideValue=0):
    """binary threshold.

    Parameters
    ----------
    img_in : sitk image
        Input image
    lowerThreshold : int, optional
        Lower threshold value.
    upperThreshold : int, optional
        Upper threshold value
    insideValue : int, optional
        Value assigned to area inside thresholds.
    outsideValue : int, optional
        Value assigned to area outside thresholds.

    Returns
    -------
    sitk image
        Filtered image
    """
    img_out = sitk.BinaryThreshold(img_in,
                                   lowerThreshold=lowerThreshold,
                                   upperThreshold=upperThreshold,
                                   insideValue=insideValue,
                                   outsideValue=outsideValue)
    return img_out


def otsu_threshold(img_in, insideValue=1, outsideValue=0):
    """otsu thresholding.

    Parameters
    ----------
    img_in : sitk image
        Input image
    insideValue : int, optional
        Description
    outsideValue : int, optional
        Description

    Returns
    -------
    sitk image
        Filtered image
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(insideValue)
    otsu_filter.SetOutsideValue(outsideValue)
    img_out = otsu_filter.Execute(img_in)

    return img_out


def confidence_connected(img_in,
                         seed,
                         numberOfIterations=1,
                         multiplier=2.5,
                         initialNeighborhoodRadius=1,
                         replaceValue=1):
    return sitk.ConfidenceConnected(
        img_in,
        seedList=[seed],
        numberOfIterations=numberOfIterations,
        multiplier=multiplier,
        initialNeighborhoodRadius=initialNeighborhoodRadius,
        replaceValue=replaceValue)


def _rescale(img, rescale: bool):
    """rescale the image if necessary.

    Parameters
    ----------
    img : sitk image
        Input image
    rescale : bool
        Rescale the image if true.

    Returns
    -------
    sitk image
        Rescaled image
    """
    if rescale:
        img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
    return img
