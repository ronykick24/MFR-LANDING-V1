GR_UNITS = {'GAPI','API','gAPI','api'}
RES_UNITS = {'OHMM','OHM.M','OHM-M','ohmm','ohm.m','ohm-m'}

def _unit(u):
    return (u or '').strip()

def find_gr_curve(las):
    for c in las.curves:
        if _unit(getattr(c,'unit','')) in GR_UNITS:
            return c.mnemonic
    for c in las.curves:
        d = (getattr(c,'descr','') or '').lower()
        if 'gamma' in d:
            return c.mnemonic
    return None

def is_resistivity_curve(unit, mnemonic, descr):
    u = _unit(unit)
    if u in RES_UNITS:
        return True
    m = (mnemonic or '').upper()
    d = (descr or '').lower()
    if m.startswith(('RP','RA','RT','RES')):
        return True
    if 'resist' in d:
        return True
    return False

def classify_mfr_mnemonic(mn):
    m = (mn or '').upper()
    kind='unknown'
    if m.startswith('RP'):
        kind='phase'
    elif m.startswith('RA'):
        kind='att'

    freq='unknown'
    if m.endswith('2'):
        freq='2MHz'
    elif m.endswith('4'):
        freq='400kHz'
    elif m.endswith(('0','1')):
        freq='100kHz'

    spacing='unknown'
    if 'RPS' in m:
        spacing='shallow'
    elif 'RPM' in m:
        spacing='medium'
    elif 'RPD' in m:
        spacing='deep'

    return {'kind':kind,'freq':freq,'spacing':spacing}
