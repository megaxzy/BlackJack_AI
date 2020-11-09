


class algo:
    #TODO  algorithm
    def get_best_action(self):
        actions=["hit","stay","drop"]
        best_actions=[]
        if self.player_sum == 21:
            return best_actions.append("stay")

        for hit_times in range(0,1): #TODO
            list_score_times=self.minmax(hit_times,self.dealer_sum_all,self.player_sum,0,0,0)  #TODO
            print(list_score_times)


    def minmax(self,hit_times,dealer_sum,player_sum,win_times,lose_times,tie_times):
        list_score_times = [0, 0, 0]
        if hit_times == 0:  #stay
            list_score_times=[0,0,0]   #win lose tie
            return self.dealer_turn(dealer_sum,list_score_times,self.cards.matrix,player_sum)
        for i in range(0,4):
            for j in range(0,52):
                if self.cards.matrix[i][j][0]==1:
                    self.cards.matrix[i][j][0] = 0
                    self.minmax(hit_times,dealer_sum,player_sum+j%13+1,win_times,lose_times,tie_times)
                    self.cards.matrix[i][j][0] = 1
        return list_score_times

    def dealer_turn(self,dealer_sum,list_score_times,temp_cards_matrix,player_sum):
        if dealer_sum >= 22:
            list_score_times[0] = list_score_times[0] + 1  # win
            return list_score_times
        if dealer_sum >= 16:
            if dealer_sum == player_sum:
                list_score_times[2] = list_score_times[2] + 1  # tie
            if dealer_sum > player_sum:
                list_score_times[1] = list_score_times[1] + 1  # lose
            if dealer_sum < player_sum:
                list_score_times[0] = list_score_times[0] + 1  # win
            return list_score_times
        temp_temp_cards_matrix = temp_cards_matrix.copy()
        for i in range(0, 1):
            for j in range(0, 52):
                if temp_cards_matrix[i][j][0] == 1:
                    #print(j)
                    temp_cards_matrix[i][j][0] = 0
                    list_score_times = self.dealer_turn(dealer_sum + j % 13 + 1, player_sum, temp_cards_matrix,list_score_times)
                    temp_cards_matrix = temp_temp_cards_matrix
        return list_score_times