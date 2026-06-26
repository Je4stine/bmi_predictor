@RestController
@RequestMapping("/api/v1/reports")
public class SalesReportController {

    private final OrderRepository orderRepository;
    private final CustomerRepository customerRepository;
    private final ProductRepository productRepository;

    public SalesReportController(
            OrderRepository orderRepository,
            CustomerRepository customerRepository,
            ProductRepository productRepository
    ) {
        this.orderRepository = orderRepository;
        this.customerRepository = customerRepository;
        this.productRepository = productRepository;
    }

    @GetMapping("/perMonth")
    public List<Map<String, Object>> getMonthlyReport(
            @RequestParam int month,
            @RequestParam int year
    ) {
        List<Order> allOrders = orderRepository.findAll();

        List<Map<String, Object>> report = new ArrayList<>();

        for (Order order : allOrders) {
            LocalDate orderDate = order.getCreatedAt().toLocalDate();

            if (orderDate.getMonthValue() == month && orderDate.getYear() == year) {

                Customer customer = customerRepository.findById(order.getCustomerId())
                        .orElse(null);

                if (customer == null) {
                    continue;
                }

                double total = 0;

                for (OrderItem item : order.getItems()) {
                    Product product = productRepository.findById(item.getProductId())
                            .orElse(null);

                    if (product != null) {
                        total += product.getPrice() * item.getQuantity();
                    }
                }

                Map<String, Object> row = new HashMap<>();
                row.put("orderId", order.getId());
                row.put("customerName", customer.getFirstName() + " " + customer.getLastName());
                row.put("customerEmail", customer.getEmail());
                row.put("orderDate", order.getCreatedAt());
                row.put("totalAmount", total);
                row.put("itemCount", order.getItems().size());

                report.add(row);
            }
        }

        report.sort((a, b) -> {
            Double totalA = (Double) a.get("totalAmount");
            Double totalB = (Double) b.get("totalAmount");
            return totalB.compareTo(totalA);
        });

        return report;
    }
}