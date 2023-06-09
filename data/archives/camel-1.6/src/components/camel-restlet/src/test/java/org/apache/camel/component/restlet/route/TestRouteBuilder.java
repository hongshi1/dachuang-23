/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.component.restlet.route;

import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.builder.RouteBuilder;

/**
 * Route builder for RestletRouteBuilderAuthTest
 * 
 * @version $Revision$
 */
public class TestRouteBuilder extends RouteBuilder {

    @Override
    public void configure() throws Exception {

        // START SNIPPET: consumer_route
        from("restlet:http://localhost:9080/securedOrders?restletMethod=post&restletRealmRef=realm").process(new Processor() {
            public void process(Exchange exchange) throws Exception {
                exchange.getOut().setBody(
                        "received [" + exchange.getIn().getBody()
                        + "] as an order id = "
                        + exchange.getIn().getHeader("id"));
            }
        });
        // END SNIPPET: consumer_route

        // START SNIPPET: producer_route
        // Note: restletMethod and restletRealmRef are stripped 
        // from the query before a request is sent as they are 
        // only processed by Camel.
        from("direct:start-auth").to("restlet:http://localhost:9080/securedOrders?restletMethod=post");
        // END SNIPPET: producer_route
      
    }

}



