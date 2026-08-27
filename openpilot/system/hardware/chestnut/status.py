import threading
import time

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, is_chestnut_usb_device
from openpilot.selfdrive.modeld.helpers import chestnut_compiled
from openpilot.system.hardware.chestnut.flash import link_up


CHESTNUT_RELEASE_BRANCHES = ("release-chestnut", "release-chestnut-staging")


class ChestnutStatus:
  def __init__(self):
    self.started = time.monotonic()
    self.last_link_check = 0.
    self.link_up: bool | None = None
    self.link_check_thread: threading.Thread | None = None
    self.link_check_generation = 0
    self.offroad = True
    self.pcie_failed = False

  def check_link(self, generation: int) -> None:
    link = link_up()
    if generation == self.link_check_generation:
      self.link_up = link

  def update(self, offroad: bool, branch: str, usb_state: list[dict], firmware_failed: bool, set_alert) -> None:
    detected = [d for d in usb_state if is_chestnut_usb_device(d["vendorId"], d["productId"], include_bootloader=True)]
    devices = [d for d in detected if is_chestnut_usb_device(d["vendorId"], d["productId"])]
    firmware_ok = len(devices) == 1 and devices[0]["product"] == CHESTNUT_USB_PRODUCT

    if self.offroad and not offroad:
      self.last_link_check = 0.
      self.link_up = None
      self.link_check_generation += 1
      self.pcie_failed = False

    if not firmware_ok:
      self.link_up = None
    elif not offroad and (self.link_check_thread is None or not self.link_check_thread.is_alive()) and time.monotonic() - self.last_link_check >= 5.:
      self.last_link_check = time.monotonic()
      self.link_check_thread = threading.Thread(target=self.check_link, args=(self.link_check_generation,), daemon=True)
      self.link_check_thread.start()

    if not offroad and self.link_up is False:
      self.pcie_failed = True

    release = branch in CHESTNUT_RELEASE_BRANCHES
    missing = offroad and release and time.monotonic() - self.started > 10. and len(detected) != 1
    slow_usb = offroad and len(devices) == 1 and devices[0]["speedMbps"] < 5000
    set_alert("Offroad_ChestnutBranch", offroad and not release and len(devices) == 1)
    set_alert("Offroad_ChestnutNotDetected", missing)
    set_alert("Offroad_ChestnutUsbSlow", slow_usb, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None)
    set_alert("Offroad_ChestnutPcieUnavailable", self.pcie_failed)
    set_alert("Offroad_ChestnutUncompiled", offroad and firmware_ok and not chestnut_compiled())
    set_alert("Offroad_ChestnutUpdateFailed", offroad and firmware_failed)
    self.offroad = offroad
