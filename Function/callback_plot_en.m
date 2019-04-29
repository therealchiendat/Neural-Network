function callback_plot_en(source,event,var_curve)
        val = source.Value;
        if val==0
            var_curve.Visible='on';
        else
            var_curve.Visible='off';
        end
    end