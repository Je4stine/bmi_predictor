<?php

declare(strict_types=1);

use PHPUnit\Framework\TestCase;

require_once __DIR__ . '/../src/PowerBankBilling.php';

class PowerBankBillingTest extends TestCase
{
    private $calculator;
    private $baseTime;

    protected function setUp(): void
    {
        $this->calculator = new PowerBankBilling();
        $this->baseTime = time();
    }

    // Helper to generate order array
    private function createOrder($durationSeconds, $billingSettings)
    {
        return [
            'start_time' => $this->baseTime,
            'end_time' => $this->baseTime + $durationSeconds,
            'billing_data' => json_encode($billingSettings)
        ];
    }

    private function getDefaultSettings($overrides = [])
    {
        return array_merge([
            'freetime' => 0,
            'billingunit' => 1, // 1 = Hour, 0 = Minute
            'billingtime' => 1,
            'amount' => 2.0,
            'ceiling' => 0,
            'deposit' => 0
        ], $overrides);
    }

    public function test_basic_hourly_billing()
    {
        // 1.5 hours = 2 billing units (rounded up). 2 units * $2 = $4
        $order = $this->createOrder(5400, $this->getDefaultSettings());
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(2, $result['unit']);
        $this->assertEquals(4.0, $result['price']);
    }

    public function test_basic_minutely_billing()
    {
        // 20 minutes. Billing unit = 15 mins. 20/15 = 2 units. 2 * $1 = $2
        $settings = $this->getDefaultSettings(['billingunit' => 0, 'billingtime' => 15, 'amount' => 1.0]);
        $order = $this->createOrder(1200, $settings); // 20 mins
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(2, $result['unit']);
        $this->assertEquals(2.0, $result['price']);
    }

    public function test_system_free_time()
    {
        // System free time: 30 mins. Usage: 45 mins. Billable: 15 mins.
        $settings = $this->getDefaultSettings([
            'freetime' => 30,
            'billingunit' => 0,
            'billingtime' => 15,
            'amount' => 1.0
        ]);
        $order = $this->createOrder(2700, $settings); // 45 mins
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(1, $result['unit']);
        $this->assertEquals(1.0, $result['price']);
        $this->assertEquals(30, $result['free_time']); // Total free time used
    }

    public function test_coupon_and_point_deductions()
    {
        // Usage: 60 mins. System Free: 10m. Coupon: 20m. Points: 15m.
        // Total Free: 45m. Billable: 15m.
        $settings = $this->getDefaultSettings([
            'freetime' => 10,
            'billingunit' => 0,
            'billingtime' => 15,
            'amount' => 2.0
        ]);
        $order = $this->createOrder(3600, $settings); // 60 mins
        $result = $this->calculator->calOrderPrice($order, 20, 15);

        $this->assertEquals(1, $result['unit']);
        $this->assertEquals(2.0, $result['price']);
        $this->assertEquals(20, $result['use_free']);
        $this->assertEquals(15, $result['use_point']);
        $this->assertEquals(45, $result['free_time']); // 10 (sys) + 20 (coupon) + 15 (point)
    }

    public function test_coupon_exceeds_billable_time()
    {
        // Usage: 10 mins. Coupon: 30 mins. Expected: 0 price, use_free capped at 10
        $settings = $this->getDefaultSettings(['billingunit' => 0, 'billingtime' => 1, 'amount' => 1.0]);
        $order = $this->createOrder(600, $settings); // 10 mins
        $result = $this->calculator->calOrderPrice($order, 30, 0);

        $this->assertEquals(0, $result['unit']);
        $this->assertEquals(0.0, $result['price']);
        $this->assertEquals(10, $result['use_free']);
    }

    public function test_daily_ceiling_under_24_hours()
    {
        // $3/hour. Ceiling $10. Usage: 5 hours. Raw: $15. Expected: $10
        $settings = $this->getDefaultSettings(['amount' => 3.0, 'ceiling' => 10.0]);
        $order = $this->createOrder(18000, $settings); // 5 hours
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(10.0, $result['price']);
    }

    public function test_daily_ceiling_over_24_hours()
    {
        // $3/hour. Ceiling $10. Usage: 26 hours (1 day + 2 hours).
        // Day 1: $10. Remainder (2h): $6. Total: $16.
        $settings = $this->getDefaultSettings(['amount' => 3.0, 'ceiling' => 10.0]);
        $order = $this->createOrder(93600, $settings); // 26 hours
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(16.0, $result['price']);
    }

    public function test_daily_ceiling_over_24_hours_with_remainder_capped()
    {
        // $3/hour. Ceiling $10. Usage: 28 hours (1 day + 4 hours).
        // Day 1: $10. Remainder (4h): $12 -> Capped at $10. Total: $20.
        $settings = $this->getDefaultSettings(['amount' => 3.0, 'ceiling' => 10.0]);
        $order = $this->createOrder(100800, $settings); // 28 hours
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(20.0, $result['price']);
    }

    public function test_deposit_cap()
    {
        // Calculated price: $10. Deposit: $5. Expected: $5
        $settings = $this->getDefaultSettings(['amount' => 10.0, 'deposit' => 5.0]);
        $order = $this->createOrder(3600, $settings); // 1 hour
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(5.0, $result['price']);
    }

    public function test_zero_or_negative_time()
    {
        // End time before start time
        $order = [
            'start_time' => $this->baseTime,
            'end_time' => $this->baseTime - 100,
            'billing_data' => json_encode($this->getDefaultSettings())
        ];
        $result = $this->calculator->calOrderPrice($order);

        $this->assertEquals(0, $result['unit']);
        $this->assertEquals(0.0, $result['price']);
    }
}
