

def interval2str(interval, fmt='{:0.1f}, {:0.1f}'):
    """converting pd.Interval data type to a more readable string"""
    range_str = fmt.format(interval.left, interval.right)
    if interval.closed == 'both':
        output_str = '[' + range_str + ']'
    elif interval.closed == 'left':
        output_str = '[' + range_str + ')'
    elif interval.closed == 'right':
        output_str = '(' + range_str + ']'
    elif interval.closed == 'neither':
        output_str = '(' + range_str + ')'
    else:
        raise ValueError('Unknown bound')
    return output_str

