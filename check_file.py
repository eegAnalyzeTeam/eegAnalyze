import os
import os.path as op
import re
import time

import numpy as np


from io import StringIO
import configparser
# from ...utils import verbose, logger, warn
# from ..constants import FIFF
# from ..meas_info import _empty_info
# from ..base import BaseRaw, _check_update_montage
# from ..utils import (_read_segments_file, _synthesize_stim_channel,
#                      _mult_cal_one)

def get_vhdr_info(vhdr_fname):
    """Extract all the information from the header file.

    Parameters
    ----------
    vhdr_fname : str
        Raw EEG header to be read.
    eog : list of str
        Names of channels that should be designated EOG channels. Names should
        correspond to the vhdr file.
    misc : list or tuple of str | 'auto'
        Names of channels or list of indices that should be designated
        MISC channels. Values should correspond to the electrodes
        in the vhdr file. If 'auto', units in vhdr file are used for inferring
        misc channels. Default is ``'auto'``.
    scale : float
        The scaling factor for EEG data. Unless specified otherwise by
        header file, units are in microvolts. Default scale factor is 1.
    montage : str | None | instance of Montage
        Path or instance of montage containing electrode positions. If None,
        read sensor locations from header file if present, otherwise (0, 0, 0).
        See the documentation of :func:`mne.channels.read_montage` for more
        information.

    Returns
    -------
    info : Info
        The measurement info.
    fmt : str
        The data format in the file.
    edf_info : dict
        A dict containing Brain Vision specific parameters.
    events : array, shape (n_events, 3)
        Events from the corresponding vmrk file.

    """
    #scale = float(scale)
    ext = op.splitext(vhdr_fname)[-1]
    if ext != '.vhdr':
        raise IOError("The header file must be given to read the data, "
                      "not a file with extension '%s'." % ext)
    with open(vhdr_fname, 'rb') as f:
        # extract the first section to resemble a cfg
        header = f.readline()
        codepage = 'utf-8'
        # we don't actually need to know the coding for the header line.
        # the characters in it all belong to ASCII and are thus the
        # same in Latin-1 and UTF-8
        header = header.decode('ascii', 'ignore').strip()
        #_check_hdr_version(header)

        settings = f.read()
        try:
            # if there is an explicit codepage set, use it
            # we pretend like it's ascii when searching for the codepage
            cp_setting = re.search('Codepage=(.+)',
                                   settings.decode('ascii', 'ignore'),
                                   re.IGNORECASE & re.MULTILINE)
            if cp_setting:
                codepage = cp_setting.group(1).strip()
            # BrainAmp Recorder also uses ANSI codepage
            # an ANSI codepage raises a LookupError exception
            # python recognize ANSI decoding as cp1252
            if codepage == 'ANSI':
                codepage = 'cp1252'
            settings = settings.decode(codepage)
        except UnicodeDecodeError:
            # if UTF-8 (new standard) or explicit codepage setting fails,
            # fallback to Latin-1, which is Windows default and implicit
            # standard in older recordings
            settings = settings.decode('latin-1')

    if settings.find('[Comment]') != -1:
        params, settings = settings.split('[Comment]')
    else:
        params, settings = settings, ''
    cfg = configparser.ConfigParser()
    if hasattr(cfg, 'read_file'):  # newer API
        cfg.read_file(StringIO(params))
    else:
        cfg.readfp(StringIO(params))

    # get sampling info
    # Sampling interval is given in microsec
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print([option for option in cfg['Common Infos']])
    print(cfg.get('Common Infos', 'MarkerFile'))
    print(cfg.get('Common Infos', 'DataFile'))
    print(cfg.get('Common Infos', 'DataFormat'))
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    # sfreq = 1e6 / cfg.getfloat('Common Infos', 'SamplingInterval')
    # info = _empty_info(sfreq)

    # order = cfg.get('Common Infos', 'DataOrientation')
    # if order not in _orientation_dict:
    #     raise NotImplementedError('Data Orientation %s is not supported'
    #                               % order)
    # order = _orientation_dict[order]

    # data_format = cfg.get('Common Infos', 'DataFormat')
    # if data_format == 'BINARY':
    #     fmt = cfg.get('Binary Infos', 'BinaryFormat')
    #     if fmt not in _fmt_dict:
    #         raise NotImplementedError('Datatype %s is not supported' % fmt)
    #     fmt = _fmt_dict[fmt]
    # else:
    #     fmt = dict((key, cfg.get('ASCII Infos', key))
    #                for key in cfg.options('ASCII Infos'))

    # # locate EEG and marker files
    # path = op.dirname(vhdr_fname)
    # data_filename = op.join(path, cfg.get('Common Infos', 'DataFile'))
    # info['meas_date'] = int(time.time())
    # info['buffer_size_sec'] = 1.  # reasonable default

    # # load channel labels
    # nchan = cfg.getint('Common Infos', 'NumberOfChannels') + 1
    # n_samples = None
    # if order == 'C':
    #     try:
    #         n_samples = cfg.getint('Common Infos', 'DataPoints')
    #     except configparser.NoOptionError:
    #         logger.warning('No info on DataPoints found. Inferring number of '
    #                        'samples from the data file size.')
    #         with open(data_filename, 'rb') as fid:
    #             fid.seek(0, 2)
    #             n_bytes = fid.tell()
    #             n_samples = n_bytes // _fmt_byte_dict[fmt] // (nchan - 1)

    # ch_names = [''] * nchan
    # cals = np.empty(nchan)
    # ranges = np.empty(nchan)
    # cals.fill(np.nan)
    # ch_dict = dict()
    # misc_chs = dict()
    # for chan, props in cfg.items('Channel Infos'):
    #     n = int(re.findall(r'ch(\d+)', chan)[0]) - 1
    #     props = props.split(',')
    #     # default to microvolts because that's what the older brainvision
    #     # standard explicitly assumed; the unit is only allowed to be
    #     # something else if explicitly stated (cf. EEGLAB export below)
    #     if len(props) < 4:
    #         props += (u'µV',)
    #     name, _, resolution, unit = props[:4]
    #     ch_dict[chan] = name
    #     ch_names[n] = name
    #     if resolution == "":
    #         if not(unit):  # For truncated vhdrs (e.g. EEGLAB export)
    #             resolution = 0.000001
    #         else:
    #             resolution = 1.  # for files with units specified, but not res
    #     unit = unit.replace(u'\xc2', u'')  # Remove unwanted control characters
    #     cals[n] = float(resolution)
    #     ranges[n] = _unit_dict.get(unit, 1) * scale
    #     if unit not in ('V', u'µV', 'uV'):
    #         misc_chs[name] = (FIFF.FIFF_UNIT_CEL if unit == 'C'
    #                           else FIFF.FIFF_UNIT_NONE)
    # misc = list(misc_chs.keys()) if misc == 'auto' else misc

    # # create montage
    # if cfg.has_section('Coordinates') and montage is None:
    #     from ...transforms import _sph_to_cart
    #     from ...channels.montage import Montage
    #     montage_pos = list()
    #     montage_names = list()
    #     to_misc = list()
    #     for ch in cfg.items('Coordinates'):
    #         ch_name = ch_dict[ch[0]]
    #         montage_names.append(ch_name)
    #         radius, theta, phi = map(float, ch[1].split(','))
    #         # 1: radius, 2: theta, 3: phi
    #         pol = np.deg2rad(theta)
    #         az = np.deg2rad(phi)
    #         pos = _sph_to_cart(np.array([[radius * 85., az, pol]]))[0]
    #         if (pos == 0).all() and ch_name not in list(eog) + misc:
    #             to_misc.append(ch_name)
    #         montage_pos.append(pos)
    #     montage_sel = np.arange(len(montage_pos))
    #     montage = Montage(montage_pos, montage_names, 'Brainvision',
    #                       montage_sel)
    #     if len(to_misc) > 0:
    #         misc += to_misc
    #         warn('No coordinate information found for channels {}. '
    #              'Setting channel types to misc. To avoid this warning, set '
    #              'channel types explicitly.'.format(to_misc))

    # ch_names[-1] = 'STI 014'
    # cals[-1] = 1.
    # ranges[-1] = 1.
    # if np.isnan(cals).any():
    #     raise RuntimeError('Missing channel units')

    # # Attempts to extract filtering info from header. If not found, both are
    # # set to zero.
    # settings = settings.splitlines()
    # idx = None

    # if 'Channels' in settings:
    #     idx = settings.index('Channels')
    #     settings = settings[idx + 1:]
    #     hp_col, lp_col = 4, 5
    #     for idx, setting in enumerate(settings):
    #         if re.match(r'#\s+Name', setting):
    #             break
    #         else:
    #             idx = None

    # # If software filters are active, then they override the hardware setup
    # # But we still want to be able to double check the channel names
    # # for alignment purposes, we keep track of the hardware setting idx
    # idx_amp = idx

    # if 'S o f t w a r e  F i l t e r s' in settings:
    #     idx = settings.index('S o f t w a r e  F i l t e r s')
    #     for idx, setting in enumerate(settings[idx + 1:], idx + 1):
    #         if re.match(r'#\s+Low Cutoff', setting):
    #             hp_col, lp_col = 1, 2
    #             warn('Online software filter detected. Using software '
    #                  'filter settings and ignoring hardware values')
    #             break
    #         else:
    #             idx = idx_amp

    # if idx:
    #     lowpass = []
    #     highpass = []

    #     # for newer BV files, the unit is specified for every channel
    #     # separated by a single space, while for older files, the unit is
    #     # specified in the column headers
    #     divider = r'\s+'
    #     if 'Resolution / Unit' in settings[idx]:
    #         shift = 1  # shift for unit
    #     else:
    #         shift = 0

    #     # Extract filter units and convert from seconds to Hz if necessary.
    #     # this cannot be done as post-processing as the inverse t-f
    #     # relationship means that the min/max comparisons don't make sense
    #     # unless we know the units.
    #     #
    #     # For reasoning about the s to Hz conversion, see this reference:
    #     # `Ebersole, J. S., & Pedley, T. A. (Eds.). (2003).
    #     # Current practice of clinical electroencephalography.
    #     # Lippincott Williams & Wilkins.`, page 40-41
    #     header = re.split(r'\s\s+', settings[idx])
    #     hp_s = '[s]' in header[hp_col]
    #     lp_s = '[s]' in header[lp_col]

    #     for i, ch in enumerate(ch_names[:-1], 1):
    #         line = re.split(divider, settings[idx + i])
    #         # double check alignment with channel by using the hw settings
    #         if idx == idx_amp:
    #             line_amp = line
    #         else:
    #             line_amp = re.split(divider, settings[idx_amp + i])
    #         assert ch in line_amp

    #         highpass.append(line[hp_col + shift])
    #         lowpass.append(line[lp_col + shift])
    #     if len(highpass) == 0:
    #         pass
    #     elif len(set(highpass)) == 1:
    #         if highpass[0] in ('NaN', 'Off'):
    #             pass  # Placeholder for future use. Highpass set in _empty_info
    #         elif highpass[0] == 'DC':
    #             info['highpass'] = 0.
    #         else:
    #             info['highpass'] = float(highpass[0])
    #             if hp_s:
    #                 # filter time constant t [secs] to Hz conversion: 1/2*pi*t
    #                 info['highpass'] = 1. / (2 * np.pi * info['highpass'])

    #     else:
    #         heterogeneous_hp_filter = True
    #         if hp_s:
    #             # We convert channels with disabled filters to having
    #             # highpass relaxed / no filters
    #             highpass = [float(filt) if filt not in ('NaN', 'Off', 'DC')
    #                         else np.Inf for filt in highpass]
    #             info['highpass'] = np.max(np.array(highpass, dtype=np.float))
    #             # Coveniently enough 1 / np.Inf = 0.0, so this works for
    #             # DC / no highpass filter
    #             # filter time constant t [secs] to Hz conversion: 1/2*pi*t
    #             info['highpass'] = 1. / (2 * np.pi * info['highpass'])

    #             # not exactly the cleanest use of FP, but this makes us
    #             # more conservative in *not* warning.
    #             if info['highpass'] == 0.0 and len(set(highpass)) == 1:
    #                 # not actually heterogeneous in effect
    #                 # ... just heterogeneously disabled
    #                 heterogeneous_hp_filter = False
    #         else:
    #             highpass = [float(filt) if filt not in ('NaN', 'Off', 'DC')
    #                         else 0.0 for filt in highpass]
    #             info['highpass'] = np.min(np.array(highpass, dtype=np.float))
    #             if info['highpass'] == 0.0 and len(set(highpass)) == 1:
    #                 # not actually heterogeneous in effect
    #                 # ... just heterogeneously disabled
    #                 heterogeneous_hp_filter = False

    #         if heterogeneous_hp_filter:
    #             warn('Channels contain different highpass filters. '
    #                  'Lowest (weakest) filter setting (%0.2f Hz) '
    #                  'will be stored.' % info['highpass'])

    #     if len(lowpass) == 0:
    #         pass
    #     elif len(set(lowpass)) == 1:
    #         if lowpass[0] in ('NaN', 'Off'):
    #             pass  # Placeholder for future use. Lowpass set in _empty_info
    #         else:
    #             info['lowpass'] = float(lowpass[0])
    #             if lp_s:
    #                 # filter time constant t [secs] to Hz conversion: 1/2*pi*t
    #                 info['lowpass'] = 1. / (2 * np.pi * info['lowpass'])

    #     else:
    #         heterogeneous_lp_filter = True
    #         if lp_s:
    #             # We convert channels with disabled filters to having
    #             # infinitely relaxed / no filters
    #             lowpass = [float(filt) if filt not in ('NaN', 'Off')
    #                        else 0.0 for filt in lowpass]
    #             info['lowpass'] = np.min(np.array(lowpass, dtype=np.float))
    #             try:
    #                 # filter time constant t [secs] to Hz conversion: 1/2*pi*t
    #                 info['lowpass'] = 1. / (2 * np.pi * info['lowpass'])

    #             except ZeroDivisionError:
    #                 if len(set(lowpass)) == 1:
    #                     # No lowpass actually set for the weakest setting
    #                     # so we set lowpass to the Nyquist frequency
    #                     info['lowpass'] = info['sfreq'] / 2.
    #                     # not actually heterogeneous in effect
    #                     # ... just heterogeneously disabled
    #                     heterogeneous_lp_filter = False
    #                 else:
    #                     # no lowpass filter is the weakest filter,
    #                     # but it wasn't the only filter
    #                     pass
    #         else:
    #             # We convert channels with disabled filters to having
    #             # infinitely relaxed / no filters
    #             lowpass = [float(filt) if filt not in ('NaN', 'Off')
    #                        else np.Inf for filt in lowpass]
    #             info['lowpass'] = np.max(np.array(lowpass, dtype=np.float))

    #             if np.isinf(info['lowpass']):
    #                 # No lowpass actually set for the weakest setting
    #                 # so we set lowpass to the Nyquist frequency
    #                 info['lowpass'] = info['sfreq'] / 2.
    #                 if len(set(lowpass)) == 1:
    #                     # not actually heterogeneous in effect
    #                     # ... just heterogeneously disabled
    #                     heterogeneous_lp_filter = False

    #         if heterogeneous_lp_filter:
    #             # this isn't clean FP, but then again, we only want to provide
    #             # the Nyquist hint when the lowpass filter was actually
    #             # calculated from dividing the sampling frequency by 2, so the
    #             # exact/direct comparison (instead of tolerance) makes sense
    #             if info['lowpass'] == info['sfreq'] / 2.0:
    #                 nyquist = ', Nyquist limit'
    #             else:
    #                 nyquist = ""
    #             warn('Channels contain different lowpass filters. '
    #                  'Highest (weakest) filter setting (%0.2f Hz%s) '
    #                  'will be stored.' % (info['lowpass'], nyquist))

    # # Creates a list of dicts of eeg channels for raw.info
    # logger.info('Setting channel info structure...')
    # info['chs'] = []
    # for idx, ch_name in enumerate(ch_names):
    #     if ch_name in eog or idx in eog or idx - nchan in eog:
    #         kind = FIFF.FIFFV_EOG_CH
    #         coil_type = FIFF.FIFFV_COIL_NONE
    #         unit = FIFF.FIFF_UNIT_V
    #     elif ch_name in misc or idx in misc or idx - nchan in misc:
    #         kind = FIFF.FIFFV_MISC_CH
    #         coil_type = FIFF.FIFFV_COIL_NONE
    #         if ch_name in misc_chs:
    #             unit = misc_chs[ch_name]
    #         else:
    #             unit = FIFF.FIFF_UNIT_NONE
    #     elif ch_name == 'STI 014':
    #         kind = FIFF.FIFFV_STIM_CH
    #         coil_type = FIFF.FIFFV_COIL_NONE
    #         unit = FIFF.FIFF_UNIT_NONE
    #     else:
    #         kind = FIFF.FIFFV_EEG_CH
    #         coil_type = FIFF.FIFFV_COIL_EEG
    #         unit = FIFF.FIFF_UNIT_V
    #     info['chs'].append(dict(
    #         ch_name=ch_name, coil_type=coil_type, kind=kind, logno=idx + 1,
    #         scanno=idx + 1, cal=cals[idx], range=ranges[idx],
    #         loc=np.full(12, np.nan),
    #         unit=unit, unit_mul=0.,  # always zero- mne manual pg. 273
    #         coord_frame=FIFF.FIFFV_COORD_HEAD))

    # # for stim channel
    # print("??????????????????????")
    # print(cfg.get('Common Infos', 'MarkerFile'))
    # print("?????????????????????????")
    # mrk_fname = op.join(path, cfg.get('Common Infos', 'MarkerFile'))
    # info._update_redundant()
    # info._check_consistency()
    # return info, data_filename, fmt, order, mrk_fname, montage, n_samples
    marker = cfg.get('Common Infos', 'MarkerFile')
    eeg = cfg.get('Common Infos', 'DataFile')
    return marker[:-5], eeg[:-4]

g = get_vhdr_info('/home/caeit/Documents/work/eeg/eegData/mdd_patient/eyeopen/njh_after_pjk_20180725_open.vhdr')
print(g)