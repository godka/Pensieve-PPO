def calculate_mpc_with_handover_exhaustive(self, agent):
    # future chunks length (try 4 if that many remaining)
    video_chunk_remain = [self.video_chunk_remain[i] for i in range(self.num_agents)]
    # last_index = self.get_total_video_chunk() - video_chunk_remain

    chunk_combo_option = []
    ho_combo_option = []
    # make chunk combination options
    for combo in itertools.product(list(range(int(BITRATE_LEVELS / BITRATE_WEIGHT))),
                                   repeat=MPC_FUTURE_CHUNK_COUNT * self.num_agents):
        chunk_combo_option.append(list([BITRATE_WEIGHT * x for x in combo]))

    # make handover combination options
    for combo in itertools.product(list(range(MPC_FUTURE_CHUNK_COUNT + 1)), repeat=self.num_agents):
        ho_combo_option.append(list(combo))

    future_chunk_length = [MPC_FUTURE_CHUNK_COUNT] * self.num_agents
    for i in range(self.num_agents):
        if video_chunk_remain[i] < MPC_FUTURE_CHUNK_COUNT:
            future_chunk_length[i] = video_chunk_remain[i]

    cur_download_bws = [self.predict_download_bw(i, True) for i in range(self.num_agents)]

    cur_sat_ids = [self.cur_sat_id[i] for i in range(self.num_agents)]
    runner_up_sat_ids = [self.get_runner_up_sat_id(i, method="harmonic-mean")[0] for i in range(self.num_agents)]

    related_sat_ids = list(set(cur_sat_ids + runner_up_sat_ids))
    num_of_sats = {}
    for sat_id in related_sat_ids:
        num_of_sats[sat_id] = self.get_num_of_user_sat(sat_id)

    start_buffers = [self.buffer_size[i] / MILLISECONDS_IN_SECOND for i in range(self.num_agents)]

    next_download_bws = []
    for agent_id in range(self.num_agents):
        for i in range(MPC_PAST_CHUNK_COUNT, 0, -1):
            self.predict_bw(runner_up_sat_ids[agent_id], agent_id, True, mahimahi_ptr=self.mahimahi_ptr[agent_id] - i,
                            plus=False)
            self.predict_bw(cur_sat_ids[agent_id], agent_id, True, mahimahi_ptr=self.mahimahi_ptr[agent_id] - i,
                            plus=False)

        tmp_next_bw = self.predict_bw(runner_up_sat_ids[agent_id], agent_id, True)
        tmp_cur_bw = self.predict_bw(cur_sat_ids[agent_id], agent_id, True)
        if cur_download_bws[agent_id] is None:
            next_download_bws.append(None)
        else:
            next_download_bws.append(cur_download_bws[agent_id] * tmp_next_bw / tmp_cur_bw)

    max_rewards = [-10000000 for _ in range(self.num_agents)]
    best_combos = [[self.last_quality[i]] for i in range(self.num_agents)]
    best_bws_sum = [-10000000]
    ho_stamps = [MPC_FUTURE_CHUNK_COUNT for _ in range(self.num_agents)]
    sat_user_nums = num_of_sats

    for ho_positions in ho_combo_option:
        future_sat_user_nums = {}

        for sat_id in sat_user_nums.keys():
            future_sat_user_nums[sat_id] = np.array([sat_user_nums[sat_id]] * MPC_FUTURE_CHUNK_COUNT)

        for idx, ho_point in enumerate(ho_positions):
            cur_sat_id = cur_sat_ids[idx]
            next_sat_id = runner_up_sat_ids[idx]
            cur_nums = future_sat_user_nums[cur_sat_id]
            next_nums = future_sat_user_nums[next_sat_id]

            cur_nums[ho_point:] = cur_nums[ho_point:] - 1
            next_nums[ho_point:] = next_nums[ho_point:] + 1

            future_sat_user_nums[cur_sat_id] = cur_nums
            future_sat_user_nums[next_sat_id] = next_nums

        for full_combo in chunk_combo_option:
            combos = []
            # Break at the end of the chunk

            for agent_id in range(self.num_agents):
                cur_combo = full_combo[
                            MPC_FUTURE_CHUNK_COUNT * agent_id: MPC_FUTURE_CHUNK_COUNT * agent_id + future_chunk_length[
                                agent_id]]
                # if cur_download_bws[agent_id] is None and cur_combo != [DEFAULT_QUALITY] * MPC_FUTURE_CHUNK_COUNT:
                #     wrong_format = True
                #     break
                if cur_download_bws[agent_id] is None:
                    combos.append([np.nan] * MPC_FUTURE_CHUNK_COUNT)
                else:
                    combos.append(cur_combo)

            rewards = []
            tmp_bws_sum = []
            for agent_id, combo in enumerate(combos):
                if combo == [np.nan] * MPC_FUTURE_CHUNK_COUNT:
                    rewards.append(np.nan)
                    continue
                curr_rebuffer_time = 0
                curr_buffer = start_buffers[agent_id]
                bitrate_sum = 0
                smoothness_diff = 0
                last_quality = self.last_quality[agent_id]
                last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain[agent_id])

                for position in range(0, len(combo)):
                    # 0, 1, 2 -> 0, 2, 4
                    chunk_quality = combo[position]
                    index = last_index + position  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                    download_time = 0

                    cur_sat_id = cur_sat_ids[agent_id]
                    next_sat_id = runner_up_sat_ids[agent_id]

                    cur_sat_user_num = 1 if sat_user_nums[cur_sat_id] == 0 \
                        else sat_user_nums[cur_sat_id]
                    next_sat_user_num = 1 if sat_user_nums[next_sat_id] == 0 \
                        else sat_user_nums[next_sat_id]

                    # cur_sat_user_num = sat_user_nums[cur_sat_id]
                    cur_future_sat_user_num = future_sat_user_nums[cur_sat_id][position]
                    # next_sat_user_num = sat_user_nums[next_sat_id]
                    next_future_sat_user_num = future_sat_user_nums[next_sat_id][position]

                    if ho_positions[agent_id] > position:
                        harmonic_bw = cur_download_bws[agent_id] * cur_sat_user_num / cur_future_sat_user_num
                    elif ho_positions[agent_id] == position:
                        harmonic_bw = next_download_bws[agent_id] * next_sat_user_num / next_future_sat_user_num
                        # Give them a penalty
                        download_time += HANDOVER_DELAY

                    else:
                        harmonic_bw = next_download_bws[agent_id] * next_sat_user_num / next_future_sat_user_num

                    tmp_bws_sum.append(harmonic_bw)

                    download_time += (self.video_size[chunk_quality][index] / B_IN_MB) \
                                     / harmonic_bw * BITS_IN_BYTE  # this is MB/MB/s --> seconds

                    if curr_buffer < download_time:
                        curr_rebuffer_time += (download_time - curr_buffer)
                        curr_buffer = 0.0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += VIDEO_CHUNCK_LEN / MILLISECONDS_IN_SECOND

                    # bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                    # smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                    bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diff += abs(
                        VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                    last_quality = chunk_quality
                # compute reward for this combination (one reward per 5-chunk combo)

                # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

                # 10~140 - 0~100 - 0~130
                rewards.append(bitrate_sum * QUALITY_FACTOR / M_IN_K - (REBUF_PENALTY * curr_rebuffer_time) \
                               - SMOOTH_PENALTY * smoothness_diff / M_IN_K)

            if np.nanmean(rewards) >= np.nanmean(max_rewards):
                best_combos = combos
                max_rewards = rewards
                ho_stamps = ho_positions
                best_bws_sum = tmp_bws_sum
            elif np.nanmean(rewards) == np.nanmean(max_rewards) and \
                    (combos[agent][0] >= best_combos[agent][0] or ho_stamps[agent] >= ho_positions[agent]
                     or np.nanmean(tmp_bws_sum) >= np.nanmean(best_bws_sum)):
                # elif np.nanmean(rewards) == np.nanmean(max_rewards) \
                #         and (rewards[agent] >= max_rewards[agent] or combos[agent][0] >= best_combos[agent][0]):
                best_combos = combos
                max_rewards = rewards
                ho_stamps = ho_positions
                best_bws_sum = tmp_bws_sum