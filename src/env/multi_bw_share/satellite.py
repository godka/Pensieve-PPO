import structlog

from env.multi_bw_share.user import User
from util.constants import SUPPORTED_SHARING, EPSILON, BIG_EPSILON
import numpy as np
import copy

SNR_THRESHOLD = 2e-8


class Satellite:
    """A base station sending data to connected UEs"""

    def __init__(self, sat_id, sat_bw, sharing_model, conn_use_log=None, data_rate_ratio_log=None):

        if conn_use_log is None:
            conn_use_log = {}
        if data_rate_ratio_log is None:
            data_rate_ratio_log = {}

        self.sat_id = sat_id
        self.sat_bw = sat_bw

        # model for sharing rate/resources among connected UEs. One of SUPPORTED_SHARING models
        self.sharing_model = sharing_model
        assert self.sharing_model in SUPPORTED_SHARING, f"{self.sharing_model=} not supported. {SUPPORTED_SHARING=}"

        # set constants for SINR and data rate calculation
        # numbers originally from https://sites.google.com/site/lteencyclopedia/lte-radio-link-budgeting-and-rf-planning
        # changed numbers to get shorter range --> simulate smaller map
        self.bw = 9e6  # in Hz?
        self.frequency = 2500  # in MHz
        self.noise = 1e-9  # in mW
        self.tx_power = 30  # in dBm (was 40)
        # self.conn_ues = []

        self.conn_use_log = conn_use_log
        self.data_rate_ratio_log = data_rate_ratio_log

        self.height = 15
        # self.data_rate_ratio = {}

        # just consider downlink for now; more interesting for most apps anyway
        self.log = structlog.get_logger(sat_id=self.sat_id)
        self.log.debug('Satellite init', sharing_model=self.sharing_model)

    def copy_satellite(self, mahimahi_ptr):
        return Satellite(self.sat_id, copy.deepcopy(self.sat_bw), self.sharing_model, self.get_conn_use_log(mahimahi_ptr),
                         self.get_data_rate_ratio_log(mahimahi_ptr))

    def __repr__(self):
        return str(self.sat_id)

    def get_bw(self):
        return self.sat_bw

    def num_conn_ues(self, mahimahi_ptr):
        return len(self.get_ue_list(mahimahi_ptr))

    @property
    def total_data_rate(self):
        """Total data rate of connections from this BS to all UEs"""
        total_rate = 0
        for ue in self.conn_ues:
            total_rate += ue.bs_dr[self]
        self.log.debug('BS total rate', total_rate=total_rate)
        return total_rate

    @property
    def total_utility(self):
        """
        Total utility summed up over all UEs connected to this BS.
        Important for multi_no_bw_share-agent reward: If the BS is idle, return 0.
        """
        return sum([ue.utility for ue in self.conn_ues])

    @property
    def avg_utility(self):
        """Avg utility of UEs connected to this BS. If the BS is idle, return 0."""
        if len(self.conn_ues) > 0:
            return np.mean([ue.utility for ue in self.conn_ues])
        return 0

    @property
    def min_utility(self):
        """Min utility of UEs connected to this BS. If the BS is idle, return max utility"""
        if len(self.conn_ues) > 0:
            return min([ue.utility for ue in self.conn_ues])
        return MAX_UTILITY

    def add_bw(self, sat_bw, user_id):
        self.sat_bw[user_id] = sat_bw

    def add_ue(self, user_id, mahimahi_ptr):
        self.log.info("Add_ue", conn_use_log=self.conn_use_log, user_id=user_id, mahimahi_ptr=mahimahi_ptr)

        # print("Add: ", user_id, self.sat_id, mahimahi_ptr)
        if mahimahi_ptr in self.conn_use_log:
            self.conn_use_log[mahimahi_ptr].append([True, user_id])
        else:
            self.conn_use_log[mahimahi_ptr] = [[True, user_id]]
        ue_list = self.get_ue_list(mahimahi_ptr)
        return ue_list
        # self.conn_ues.append(user_id)

        """
        if user_id not in self.data_rate_ratio.keys():
            for tmp_id in self.data_rate_ratio.keys():
                self.data_rate_ratio[tmp_id] = (1 - 1/ self.num_conn_ues) * self.data_rate_ratio[tmp_id]

            self.data_rate_ratio[user_id] = 1 / self.num_conn_ues
        assert sum(self.data_rate_ratio.values()) <= 1
        """

    def remove_ue(self, user_id, mahimahi_ptr):
        self.log.debug("remove_ue", conn_use_log=self.conn_use_log, user_id=user_id, mahimahi_ptr=mahimahi_ptr)
        # print("Remove: ", user_id, self.sat_id, mahimahi_ptr)
        # assert user_id in self.conn_ues
        # self.conn_ues.remove(user_id)
        if mahimahi_ptr in self.conn_use_log.keys():
            self.conn_use_log[mahimahi_ptr].append([False, user_id])
        else:
            self.conn_use_log[mahimahi_ptr] = [[False, user_id]]
        self.get_ue_list(mahimahi_ptr)
        """
        if user_id in self.data_rate_ratio.keys():
            removed_ratio = self.data_rate_ratio.pop(user_id)
            for tmp_id in self.data_rate_ratio.keys():
                self.data_rate_ratio[tmp_id] += removed_ratio / len(self.data_rate_ratio.keys())
        assert sum(self.data_rate_ratio.values()) <= 1
        """

    def get_ue_list(self, mahimahi_ptr):
        ue_list = []

        for i in sorted(self.conn_use_log.keys()):
            if mahimahi_ptr < i:
                break
            for elem in self.conn_use_log[i]:
                if elem[0]:
                    assert elem[1] not in ue_list
                    ue_list.append(elem[1])
                else:
                    assert elem[1] in ue_list
                    ue_list.remove(elem[1])

        return ue_list

    def get_data_rate_ratio_log(self, mahimahi_ptr):
        recent_log = {}
        for i in sorted(self.data_rate_ratio_log.keys()):
            if mahimahi_ptr < i:
                break
            recent_log[i] = self.data_rate_ratio_log[i]

        return recent_log

    def get_conn_use_log(self, mahimahi_ptr):
        recent_log = {}
        for i in sorted(self.conn_use_log.keys()):
            if mahimahi_ptr < i:
                break
            recent_log[i] = self.conn_use_log[i]

        return recent_log

    def get_cur_data_rate(self, mahimahi_ptr):
        self.log.debug("rate", log=self.data_rate_ratio_log, ptr=mahimahi_ptr)
        recent_log = {}
        for i in sorted(self.data_rate_ratio_log.keys()):
            if mahimahi_ptr < i:
                break
            recent_log = self.data_rate_ratio_log[i]

        return recent_log

    def set_data_rate_ratio(self, user_id, ratio_list, mahimahi_ptr):
        data_rate_ratio = {}
        index = 0
        remained_ratio = 0
        for uid in user_id:
            if ratio_list[index] < EPSILON + BIG_EPSILON:
                remained_ratio += ratio_list[index]
            else:
                data_rate_ratio[uid] = ratio_list[index]
            index += 1
        for uid in data_rate_ratio.keys():
            data_rate_ratio[uid] += remained_ratio / len(data_rate_ratio.keys())

        self.data_rate_ratio_log[mahimahi_ptr] = data_rate_ratio
        self.log.info("set_data", user_id=user_id, ratio_list=ratio_list, data_rate_ratio=data_rate_ratio)
        return data_rate_ratio

    def path_loss(self, distance, ue_height=1.5):
        """Return path loss in dBm to a UE at a given position. Calculation using Okumura Hata, suburban indoor"""
        ch = 0.8 + (1.1 * np.log10(self.frequency) - 0.7) * ue_height - 1.56 * np.log10(self.frequency)
        const1 = 69.55 + 26.16 * np.log10(self.frequency) - 13.82 * np.log10(self.height) - ch
        const2 = 44.9 - 6.55 * np.log10(self.height)
        # add small epsilon to avoid log(0) if distance = 0
        return const1 + const2 * np.log10(distance + EPSILON)

    def received_power(self, distance):
        """Return the received power at a given distance"""
        return 10 ** ((self.tx_power - self.path_loss(distance)) / 10)

    def snr(self, distance):
        """Return the signal-to-noise (SNR) ratio given a UE position."""
        signal = self.received_power(distance)
        self.log.debug('SNR to UE', distance=str(distance), signal=signal)
        return signal / self.noise

    def data_rate_unshared(self, mahimahi_ptr, user: User):
        """
        Return the achievable data rate for a given UE assuming that it gets the BS' full, unshared data rate.

        :param mahimahi_ptr:
        :param user:
        :return: Return the max. achievable data rate for the UE if it were/is connected to the BS.
        """
        # snr = self.snr(distance)
        # dr_ue_unshared = self.bw * np.log2(1 + snr)
        # For test

        dr_ue_unshared = self.sat_bw[mahimahi_ptr]
        dr_ue_unshared *= user.get_snr_noise()
        # dr_ue_unshared *= np.random.uniform(SNR_NOISE_LOW, SNR_NOISE_HIGH)
        return dr_ue_unshared

    def data_rate_shared(self, user: User, dr_ue_unshared, mahimahi_ptr, plus=False):
        """
        Return the shared data rate the given UE would get based on its unshared data rate and a sharing model.

        param distance: UE requesting the achievable data rate
        :param user:
        :param dr_ue_unshared: The UE's unshared achievable data rate
        :return: The UE's final, shared data rate that it (could/does) get from this BS
        """
        assert self.sharing_model in SUPPORTED_SHARING, f"{self.sharing_model=} not supported. {SUPPORTED_SHARING=}"
        dr_ue_shared = None
        agent_id = user.get_agent_id()

        num_conn_ues = self.num_conn_ues(mahimahi_ptr)
        assert plus is False
        if plus:
            num_conn_ues += 1

        if num_conn_ues == 0:
            num_conn_ues = 1

        # resource-fair = time/bandwidth-fair: split time slots/bandwidth/RBs equally among all connected UEs
        if self.sharing_model == 'resource-fair':
            dr_ue_shared = dr_ue_unshared / num_conn_ues
        elif self.sharing_model == 'ratio-based':
            # split data rate by all already connected UEs incl. this UE
            # assert agent_id in self.data_rate_ratio
            data_rate_ratio = self.get_cur_data_rate(mahimahi_ptr)
            if agent_id not in data_rate_ratio.keys() or set(data_rate_ratio.keys()) != set(
                    self.get_ue_list(mahimahi_ptr)):
                dr_ue_shared = dr_ue_unshared / num_conn_ues
            else:
                if len(data_rate_ratio.keys()) < num_conn_ues:
                    assert False
                    dr_ue_unshared -= dr_ue_unshared / num_conn_ues * (num_conn_ues - len(data_rate_ratio.keys()))
                    dr_ue_shared = dr_ue_unshared * data_rate_ratio[agent_id]

                elif len(data_rate_ratio.keys()) == num_conn_ues:
                    dr_ue_shared = dr_ue_unshared * data_rate_ratio[agent_id]
                else:
                    assert False
                    more_ratio = 0
                    for user_id in data_rate_ratio.keys():
                        if user_id not in self.conn_ues and user_id != agent_id:
                            more_ratio += data_rate_ratio[user_id]
                    dr_ue_shared = dr_ue_unshared * (data_rate_ratio[agent_id] + more_ratio / num_conn_ues)
        else:
            print("Wrong")
            exit(1)
        return dr_ue_shared

    def data_rate(self, user: User, mahimahi_ptr, plus=False):
        """
        Return the achievable data rate for a given UE (may or may not be connected already).
        Share & split the achievable data rate among all connected UEs, pretending this UE is also connected.
        param distance: distance
        :return: Return the max. achievable data rate for the UE if it were/is connected to the BS.
        """
        # 0 data rate if the UE cannot connect because the SNR is too low
        # achievable data rate if it wasn't shared with any other connected UEs
        if mahimahi_ptr < 0:
            mahimahi_ptr = 0
        dr_ue_unshared = self.data_rate_unshared(mahimahi_ptr, user)
        # final, shared data rate depends on sharing model
        dr_ue_shared = self.data_rate_shared(user, dr_ue_unshared, mahimahi_ptr, plus)

        self.log.debug('Achievable data rate', mahimahi_ptr=mahimahi_ptr, dr_ue_unshared=dr_ue_unshared, dr_ue_shared=dr_ue_shared,
                       num_conn_ues=self.num_conn_ues(mahimahi_ptr), user_list=self.get_ue_list(mahimahi_ptr))
        return dr_ue_shared

    def is_visible(self, mahimahi_ptr):
        return True if self.sat_bw[mahimahi_ptr] != 0 else False

