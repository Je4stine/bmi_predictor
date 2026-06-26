<?php

declare(strict_types=1);

// Holds the billing logic under test.
class PowerBankBilling
{
    public function calOrderPrice($order, $free_time = 0, $point_time = 0)
    {
        $set = json_decode($order['billing_data'], true);
        $end_time = $order['end_time'];
        $time = $end_time - $order['start_time'];
        if ($set['freetime'] != 0) {
            $time = $time - $set['freetime'] * 60;
        }

        $minute = ceil($time / 60);
        $minute < 1 && $minute = 0;
        if ($free_time > 0 && $minute > 0) {
            $free_time = $minute > $free_time ? $free_time : $minute;
        } else {
            $free_time = 0;
        }

        $free = $free_time;
        $point = 0;
        if ($point_time > 0 && $minute > $free_time) {
            $m = $minute - $free_time;
            $point = $m > $point_time ? $point_time : $m;
            $free_time = $free_time + $point;
        }

        $time = $time - $free_time * 60;

        $base = ($set['billingunit'] == 1) ? 3600 : 60;
        $billingunit = 0;
        $amount = 0;
        if ($time > 0) {
            $onedayamount = (86400 / ($base * $set['billingtime'])) * $set['amount'];
            if ($set['ceiling'] > 0 && $onedayamount > $set['ceiling']) {
                if ($time > 86400) {
                    $billingseconds = $time % 86400;
                    $billingunit = ceil($billingseconds / ($base * $set['billingtime']));
                    $amountseconds = $billingunit * $set['amount'];
                    if ($amountseconds > $set['ceiling']) {
                        $amountseconds = $set['ceiling'];
                    }
                    $dayamount = (($time - $billingseconds) / 86400) * $set['ceiling'];
                    $amount = $dayamount + $amountseconds;
                } else {
                    $billingunit = ceil($time / ($base * $set['billingtime']));
                    $amount = $billingunit * $set['amount'];
                    if ($amount > $set['ceiling']) {
                        $amount = $set['ceiling'];
                    }
                }
            } else {
                $billingunit = ceil($time / ($base * $set['billingtime']));
                $amount = $billingunit * $set['amount'];
            }
        }
        if (isset($set['deposit']) && $amount > $set['deposit']) {
            $amount = $set['deposit'];
        }

        $amount < 0.01 && $amount = 0;

        return ['unit' => $billingunit, 'price' => floatval($amount), 'free_time' => $set['freetime'] + $free_time, 'use_free' => $free, 'use_point' => $point];
    }
}
