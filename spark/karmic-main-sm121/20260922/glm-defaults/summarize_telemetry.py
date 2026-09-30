#!/usr/bin/env python3
"""Summarize retained observer samples for an explicit benchmark window."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('An explicit timezone is required')
    return result


def samples(text):
    current = None
    for line in text.splitlines():
        if re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', line):
            if current is not None:
                yield current
            current = {'time': timestamp(line), 'counters': {}}
        elif current is not None:
            gpu = re.fullmatch(r'([\d.]+) MHz, (0x[\da-fA-F]+), ([\d.]+), ([\d.]+) W', line)
            memory = re.fullmatch(r'(MemAvailable|SwapFree):\s+(\d+) kB', line)
            counter = re.fullmatch(r'((?:allocstall|compact_stall|pgscan_direct)\w*) (\d+)', line)
            if gpu:
                current.update(clock=float(gpu[1]), flags=int(gpu[2], 16),
                               temperature=float(gpu[3]), power=float(gpu[4]))
            elif memory:
                current[memory[1]] = int(memory[2])
            elif counter:
                current['counters'][counter[1]] = int(counter[2])
    if current is not None:
        yield current


def summarize(text, start, end):
    if end <= start:
        raise ValueError('Invalid observation window')
    rows = [s for s in samples(text) if start <= s['time'] <= end]
    if not rows or any(not {'clock', 'flags', 'temperature', 'power', 'MemAvailable', 'SwapFree'} <= s.keys() for s in rows):
        raise ValueError('Missing or incomplete telemetry in the requested window')
    first, last = rows[0], rows[-1]
    if first['counters'].keys() != last['counters'].keys():
        raise ValueError('Counter inventory changed')
    deltas = {key: last['counters'][key] - value for key, value in first['counters'].items()}
    if not deltas or any(value < 0 for value in deltas.values()):
        raise ValueError('Counters missing or reset during the window')
    gaps = [(b['time'] - a['time']).total_seconds() for a, b in zip(rows, rows[1:])]
    return {'sample_count': len(rows), 'first_sample': first['time'].isoformat(),
            'last_sample': last['time'].isoformat(), 'max_sample_gap_seconds': max(gaps, default=0),
            'start_gap_seconds': (first['time'] - start).total_seconds(),
            'end_gap_seconds': (end - last['time']).total_seconds(),
            'mean_sm_clock_mhz': mean(s['clock'] for s in rows),
            'nonzero_clock_event_samples': sum(s['flags'] != 0 for s in rows),
            'max_temperature_c': max(s['temperature'] for s in rows),
            'mean_power_w': mean(s['power'] for s in rows),
            'min_available_gib': min(s['MemAvailable'] for s in rows) / 1048576,
            'swap_free_change_mib': (last['SwapFree'] - first['SwapFree']) / 1024,
            'vm_counter_deltas': deltas}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--start', required=True, type=timestamp)
    parser.add_argument('--end', required=True, type=timestamp)
    args = parser.parse_args()
    print(json.dumps({node: summarize((args.receipt / f'{node}-observer.log').read_text(), args.start, args.end)
                      for node in ('sparky', 'buddy', 'rocky', 'lucky')}, indent=2))
