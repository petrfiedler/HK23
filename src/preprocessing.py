import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def drop_columns_file(data):
    return data.drop(columns=['Unnamed: 0', 'id', 'filename', 'extra', 'datetime', 'file_path', 'name', 'typef', 'dir_type', 'file_type', 'is_allocated', 'is_allocated0', 'sha_256', 'epochtime', 'hour', 'minute'])


def drop_columns_event(data):
    return data.drop(columns=['Unnamed: 0', 'execution_process_id', 'keywords_wdi_context', 'audit_detailed_tracking', 'id', 'datetime', 'computer_name', 'source_name', 'filename_type', 'user_sid', 'channel', 'keywords', 'epochtime', 'hour', 'minute'])


def retype_file(data):
    bools = [
        'M', 'A', 'C', 'B',
        'file_stat', 'NTFS_file_stat', 'file_entry_shell_item', 'NTFS_USN_change',
        'filef', 'directory', 'link',
        'dir_appdata', 'dir_win', 'dir_user', 'dir_other',
        'file_executable', 'file_graphic', 'file_documents', 'file_ps', 'file_other',
        'mft', 'lnk_shell_items', 'olecf_olecf_automatic_destinations/lnk/shell_items', 'winreg_bagmru/shell_items', 'usnjrnl',
        'is_allocated1'
    ]

    data[bools] = data[bools].astype('boolean')

    data['timestamp'] = data['timestamp'].astype('datetime64[ns]')

    return data


def retype_event(data):
    bools = [
        'filename_security',
        'filename_application',
        'filename_system',
        'filename_rdp',
        'filename_powershell',
        'filename_other',
        'recovered',
        'audit_account_logon',
        'audit_account_management',
        'audit_logon_logoff',
        'audit_ds_access',
        'audit_object_access',
        'audit_policy_change',
        'audit_privilege_usage',
        'audit_system',
        'keywords_audit_failure',
        'keywords_audit_success',
        'keywords_correlation_hint',
        'keywords_event_log_classic',
        'keywords_sqm',
        'keywords_wdi_diagnostic'
    ]

    data[bools] = data[bools].astype('boolean')

    data['timestamp'] = data['timestamp'].astype('datetime64[ns]')

    return data


def process_file_size(data):
    data['file_size+1_log'] = np.log(data['file_size'] + 1)

    size_stamps = [-float('inf'), 0, 1_000, 10_000,
                   100_000, 1_000_000, float('inf')]
    data['file_size'] = pd.cut(
        data['file_size'], size_stamps, labels=False)

    return data


def get_minute_frequency(data):
    floored_to_min = data['timestamp'].dt.floor('min')
    counts = floored_to_min.value_counts().to_dict()
    minute_activity = data['timestamp'].apply(lambda x: counts[x.floor('min')])
    data['minute_activity'] = minute_activity

    return data


def process_inodes(data):
    inode_counts = data['inode'].value_counts().to_dict()
    inode_activity = data['inode'].apply(lambda x: inode_counts[x])
    data['inode_activity'] = inode_activity
    return data.drop(columns=['inode', 'timestamp'])


def standardize(data):
    standard_scaler = StandardScaler()
    return pd.DataFrame(standard_scaler.fit_transform(data), columns=data.columns)


def preprocess_file(data):
    data = drop_columns_file(data)
    data = retype_file(data)
    data = process_file_size(data)
    data = get_minute_frequency(data)
    data = process_inodes(data)
    data = standardize(data)
    return data


def preprocess_event(data):
    data = drop_columns_event(data)
    data = retype_event(data)
    data = get_minute_frequency(data)
    data = process_inodes(data)
    data = standardize(data)
    return data
