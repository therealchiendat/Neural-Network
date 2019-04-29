function [var_xi_val,var_classerror]=...
    function_validation(var_w1,var_w2,var_epoch,patterns_v,trueclass_v,var_d,var_n,var_sumlay1)
    for var_count=1:120
        %Layer 1
        for var_i=1:var_n
            var_sumlay1(var_i)=var_w1(1,var_i,var_epoch)...
                +patterns_v(1,var_count)*var_w1(2,var_i,var_epoch)...
                +patterns_v(2,var_count)*var_w1(3,var_i,var_epoch);
        end
        var_lay1 =tanh(var_sumlay1);
        %Layer 2
        var_sumlay2=[1,var_lay1             ]*var_w2(:,:,var_epoch);
        var_lay2 =tanh(var_sumlay2);
        %Output error
        var_d(trueclass_v(var_count))=1;
        var_error_val(:,var_count)  =(var_d-var_lay2);
        var_find_trueclass(var_count) =isequal(find(var_lay2==max(var_lay2)),trueclass_v(var_count));
    end
    %find ASSE
    var_xi_val=mean(sum(var_error_val.^2,1));
    %find classification error
    
    var_classerror=sum(var_find_trueclass);
end